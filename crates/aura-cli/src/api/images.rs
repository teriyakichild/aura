//! Local image files attached to a user message.
//!
//! An attachment is read from disk once, base64-encoded, and carried as the
//! `data:` URL the server's `image_url` content part expects. The media type
//! comes from the file extension; only the formats vision providers accept
//! as inline base64 are allowed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use crate::api::types::ImageUrl;

/// A local image loaded for a chat turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAttachment {
    /// Display name: the file name of the loaded path.
    pub name: String,
    pub media_type: &'static str,
    /// Size of the file on disk, in bytes.
    pub bytes: u64,
    /// `data:<media_type>;base64,<payload>`.
    pub data_url: String,
}

/// Map a file extension to the media type sent to the server.
///
/// The list is the intersection of what rig models (`ImageMediaType`) and
/// what OpenAI, Anthropic, Bedrock, Gemini and Ollama accept as inline base64.
fn media_type_for_extension(extension: &str) -> Option<&'static str> {
    match extension.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Expand a leading `~` or `~/` to the user's home directory.
fn expand_home(path: &str) -> PathBuf {
    if (path == "~" || path.starts_with("~/"))
        && let Some(home) = dirs::home_dir()
    {
        return home.join(path.trim_start_matches('~').trim_start_matches('/'));
    }
    PathBuf::from(path)
}

impl ImageAttachment {
    /// Read `path` from disk and encode it for an `image_url` part.
    ///
    /// Fails when the file cannot be read or its extension is not a supported
    /// image format; the message names the path so the user can fix the
    /// command line.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default();
        let Some(media_type) = media_type_for_extension(extension) else {
            bail!(
                "{}: unsupported image type (expected .png, .jpg, .jpeg, .gif or .webp)",
                path.display()
            );
        };
        let bytes = std::fs::read(path)
            .with_context(|| format!("{}: cannot read image", path.display()))?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        Ok(Self {
            name,
            media_type,
            bytes: bytes.len() as u64,
            data_url: format!("data:{media_type};base64,{}", STANDARD.encode(&bytes)),
        })
    }

    /// [`Self::load`] after `~` expansion of a path typed at a prompt.
    pub fn load_user_path(path: &str) -> Result<Self> {
        Self::load(expand_home(path))
    }

    /// The `image_url` part for this attachment. `detail` is left unset so the
    /// server applies the provider's default fidelity.
    pub fn to_image_url(&self) -> ImageUrl {
        ImageUrl {
            url: self.data_url.clone(),
            detail: None,
        }
    }

    /// `name (size)` for echo lines and error messages.
    pub fn label(&self) -> String {
        format!("{} ({})", self.name, human_size(self.bytes))
    }
}

/// Load every path in order, stopping at the first failure.
pub fn load_all<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<ImageAttachment>> {
    paths.iter().map(ImageAttachment::load).collect()
}

fn human_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    let b = bytes as f64;
    if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.0} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_fixture(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();
        path
    }

    #[test]
    fn load_encodes_png_as_data_url() {
        let dir = tempfile::tempdir().unwrap();
        let path = png_fixture(&dir, "shot.PNG");
        let image = ImageAttachment::load(&path).unwrap();
        assert_eq!(image.name, "shot.PNG");
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.bytes, 8);
        assert_eq!(image.data_url, "data:image/png;base64,iVBORw0KGgo=");
        assert_eq!(image.to_image_url().url, image.data_url);
        assert_eq!(image.to_image_url().detail, None);
        assert_eq!(image.label(), "shot.PNG (8 B)");
    }

    #[test]
    fn load_maps_jpeg_extensions() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.jpg", "b.jpeg", "c.JPEG"] {
            let path = png_fixture(&dir, name);
            assert_eq!(
                ImageAttachment::load(&path).unwrap().media_type,
                "image/jpeg",
                "{name}"
            );
        }
    }

    #[test]
    fn load_rejects_unsupported_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = png_fixture(&dir, "notes.txt");
        let err = ImageAttachment::load(&path).unwrap_err().to_string();
        assert!(err.contains("notes.txt"), "{err}");
        assert!(err.contains("unsupported image type"), "{err}");
    }

    #[test]
    fn load_reports_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.png");
        let err = format!("{:#}", ImageAttachment::load(&path).unwrap_err());
        assert!(err.contains("missing.png"), "{err}");
        assert!(err.contains("cannot read image"), "{err}");
    }

    #[test]
    fn load_all_stops_at_first_failure() {
        let dir = tempfile::tempdir().unwrap();
        let good = png_fixture(&dir, "ok.png");
        let bad = dir.path().join("nope.png");
        let err = load_all(&[good.clone(), bad]).unwrap_err().to_string();
        assert!(err.contains("nope.png"), "{err}");
        assert_eq!(load_all(&[good]).unwrap().len(), 1);
    }

    #[test]
    fn expand_home_only_touches_leading_tilde() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_home("~/x.png"), home.join("x.png"));
        assert_eq!(expand_home("~"), home);
        assert_eq!(expand_home("a/~/x.png"), PathBuf::from("a/~/x.png"));
        assert_eq!(expand_home("/abs/x.png"), PathBuf::from("/abs/x.png"));
    }

    #[test]
    fn human_size_picks_unit() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(2048), "2 KiB");
        assert_eq!(human_size(3 * 1024 * 1024), "3.0 MiB");
    }
}
