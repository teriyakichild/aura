//! Tab completion for the `/image <path>` argument.
//!
//! Completion is shell-like. The candidates for a typed prefix are the
//! longest common extension of the matching names (when it adds characters)
//! followed by each match, so Tab first extends the name as far as every
//! match agrees, then cycles through the matches one by one, and finally
//! restores what was typed. At the prompt rustyline's circular completion
//! drives that cycle through [`image_completions`]; while a response streams
//! the mid-stream key loop drives it through [`image_tab`]. Only the first
//! argument completes; once a message follows the path, Tab does nothing.

use std::sync::Mutex;

use crate::api::images::{expand_home, media_type_for_extension};

/// Upper bound on directory entries offered, so a huge folder stays cheap to
/// list on every keystroke.
const MAX_MATCHES: usize = 500;

/// The path argument typed so far, while the cursor is still inside it.
///
/// Returns `None` for lines that are not `/image`, and for lines where a
/// message already follows the path (a second word, or a closed quote).
pub(crate) fn image_path_prefix(line: &str) -> Option<&str> {
    let arg = match line.strip_prefix("/image") {
        Some("") => "",
        Some(rest) if rest.starts_with(' ') => rest.trim_start(),
        _ => return None,
    };
    if let Some(body) = arg.strip_prefix('"').or_else(|| arg.strip_prefix('\'')) {
        let quote = arg.chars().next().unwrap_or('"');
        return (!body.contains(quote)).then_some(body);
    }
    (!arg.contains(char::is_whitespace)).then_some(arg)
}

/// Split a path prefix into the directory part (up to and including the last
/// `/`, possibly empty) and the partial name after it.
fn split_dir_name(prefix: &str) -> (&str, &str) {
    match prefix.rfind('/') {
        Some(idx) => (&prefix[..=idx], &prefix[idx + 1..]),
        None => ("", prefix),
    }
}

/// Entries under the prefix's directory that continue its partial name:
/// sub-directories (with a trailing `/`) and image files. Hidden entries are
/// offered only when the partial name itself starts with a dot. Directories
/// sort before files, each alphabetically.
pub(crate) fn image_path_matches(prefix: &str) -> Vec<String> {
    let (dir_part, name_part) = split_dir_name(prefix);
    let dir = if dir_part.is_empty() {
        std::path::PathBuf::from(".")
    } else {
        expand_home(dir_part)
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let wanted = name_part.to_lowercase();
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in entries.flatten().take(MAX_MATCHES) {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if name.starts_with('.') && !name_part.starts_with('.') {
            continue;
        }
        if !name.to_lowercase().starts_with(&wanted) {
            continue;
        }
        if entry.path().is_dir() {
            dirs.push(format!("{name}/"));
        } else if media_type_for_extension(extension_of(&name)).is_some() {
            files.push(name);
        }
    }
    dirs.sort();
    files.sort();
    dirs.extend(files);
    dirs
}

/// The argument text that puts `name` under `prefix`'s directory. A
/// completed file gets a trailing space so the message can be typed next; a
/// directory (trailing `/`) or a partial name leaves the cursor at the end of
/// the path. Paths containing whitespace are quoted; the quote is closed only
/// after a file.
pub(crate) fn path_argument(prefix: &str, name: &str) -> String {
    let (dir_part, _) = split_dir_name(prefix);
    let path = format!("{dir_part}{name}");
    let is_dir = name.ends_with('/');
    let is_file = !is_dir && media_type_for_extension(extension_of(name)).is_some();
    let quoted = path.contains(char::is_whitespace);
    match (quoted, is_file) {
        (true, true) => format!("\"{path}\" "),
        (true, false) => format!("\"{path}"),
        (false, true) => format!("{path} "),
        (false, false) => path,
    }
}

fn extension_of(name: &str) -> &str {
    std::path::Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
}

/// Longest prefix shared by every name, compared case-insensitively so that
/// mixed-case siblings still extend the typed text as far as they agree.
fn common_prefix(names: &[String]) -> String {
    let Some(first) = names.first() else {
        return String::new();
    };
    let mut end = first.len();
    for name in &names[1..] {
        end = first
            .char_indices()
            .zip(name.chars())
            .take_while(|((_, a), b)| a.eq_ignore_ascii_case(b))
            .map(|((idx, ch), _)| idx + ch.len_utf8())
            .last()
            .unwrap_or(0)
            .min(end);
    }
    first[..end].to_string()
}

/// Replacement candidates for the path argument `prefix`, in Tab order:
/// the common extension of several matches when it adds characters, then
/// every match. Empty when nothing matches.
pub(crate) fn candidates_for(prefix: &str, matches: &[String]) -> Vec<String> {
    let (_, name_part) = split_dir_name(prefix);
    let mut candidates = Vec::with_capacity(matches.len() + 1);
    if matches.len() > 1 {
        let extension = common_prefix(matches);
        if extension.len() > name_part.len() {
            candidates.push(path_argument(prefix, &extension));
        }
    }
    candidates.extend(matches.iter().map(|name| path_argument(prefix, name)));
    candidates
}

/// Offset in `line` where the path argument `prefix` starts, stepping back
/// over an opening quote so a replacement covers it.
fn argument_start(line: &str, prefix: &str) -> usize {
    let head = &line[..line.len() - prefix.len()];
    if head.ends_with(['"', '\'']) {
        head.len() - 1
    } else {
        head.len()
    }
}

/// Where the path argument starts in `line` (the opening quote included)
/// and the candidates that may replace it, for a cursor at the end of the
/// line. `None` when the line is not an `/image` path argument.
pub(crate) fn image_completions(line: &str, pos: usize) -> Option<(usize, Vec<String>)> {
    if pos != line.len() {
        return None;
    }
    let prefix = image_path_prefix(line)?;
    let start = argument_start(line, prefix);
    Some((start, candidates_for(prefix, &image_path_matches(prefix))))
}

/// A mid-stream Tab cycle in progress.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImageCycle {
    /// The line before the first Tab.
    original: String,
    /// Offset in `original` where the path argument starts.
    start: usize,
    candidates: Vec<String>,
    /// Index into `candidates`; `candidates.len()` shows `original` again.
    index: usize,
    /// The line the last Tab wrote.
    produced: String,
}

/// Outcome of a Tab press on the mid-stream buffer.
#[derive(Debug, PartialEq)]
pub(crate) enum TabStep {
    /// The line is not an `/image` path argument.
    NotImage,
    /// Nothing to complete.
    NoMatch,
    /// Replace the buffer with this text.
    Line(String),
}

/// One Tab press on `line`. `cycle` is the state the previous press left;
/// the returned state replaces it. Mirrors rustyline's circular completion:
/// candidates in order, then the original text, then around again.
pub(crate) fn tab_step(
    line: &str,
    forward: bool,
    cycle: Option<ImageCycle>,
    matches_for: impl Fn(&str) -> Vec<String>,
) -> (TabStep, Option<ImageCycle>) {
    let mut cycle = match cycle.filter(|cycle| cycle.produced == line) {
        Some(cycle) => cycle,
        None => {
            let Some(prefix) = image_path_prefix(line) else {
                return (TabStep::NotImage, None);
            };
            let candidates = candidates_for(prefix, &matches_for(prefix));
            if candidates.is_empty() {
                return (TabStep::NoMatch, None);
            }
            ImageCycle {
                original: line.to_string(),
                start: argument_start(line, prefix),
                candidates,
                // One past the end is the original; the step below moves
                // off it in the requested direction.
                index: usize::MAX,
                produced: String::new(),
            }
        }
    };
    let slots = cycle.candidates.len() + 1;
    cycle.index = if cycle.index == usize::MAX {
        if forward {
            0
        } else {
            cycle.candidates.len() - 1
        }
    } else if forward {
        (cycle.index + 1) % slots
    } else {
        (cycle.index + slots - 1) % slots
    };
    cycle.produced = match cycle.candidates.get(cycle.index) {
        Some(candidate) => format!("{}{candidate}", &cycle.original[..cycle.start]),
        None => cycle.original.clone(),
    };
    (TabStep::Line(cycle.produced.clone()), Some(cycle))
}

static CYCLE: Mutex<Option<ImageCycle>> = Mutex::new(None);

/// Tab on the mid-stream buffer: advance the shared cycle.
pub(crate) fn image_tab(line: &str, forward: bool) -> TabStep {
    let Ok(mut guard) = CYCLE.lock() else {
        return TabStep::NoMatch;
    };
    let (step, next) = tab_step(line, forward, guard.take(), image_path_matches);
    *guard = next;
    step
}

/// Names to list under the frame for an `/image` line, or `None` once a
/// message follows the path.
pub(crate) fn image_hint(line: &str) -> Option<Vec<String>> {
    image_path_prefix(line).map(image_path_matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_tracks_the_first_argument_only() {
        assert_eq!(image_path_prefix("/image"), Some(""));
        assert_eq!(image_path_prefix("/image "), Some(""));
        assert_eq!(image_path_prefix("/image ~/Pic"), Some("~/Pic"));
        assert_eq!(image_path_prefix("/image \"my d"), Some("my d"));
        assert_eq!(image_path_prefix("/image 'my d"), Some("my d"));
        assert_eq!(image_path_prefix("/image a.png hi"), None);
        assert_eq!(image_path_prefix("/image \"a b.png\" hi"), None);
        assert_eq!(image_path_prefix("/image \"a b.png\""), None);
        assert_eq!(image_path_prefix("/imagex"), None);
        assert_eq!(image_path_prefix("/model"), None);
        assert_eq!(image_path_prefix("hello"), None);
    }

    #[test]
    fn split_dir_name_keeps_the_slash_on_the_dir() {
        assert_eq!(split_dir_name(""), ("", ""));
        assert_eq!(split_dir_name("shot"), ("", "shot"));
        assert_eq!(split_dir_name("~/Pictures/"), ("~/Pictures/", ""));
        assert_eq!(split_dir_name("a/b/c.png"), ("a/b/", "c.png"));
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "Alpha.PNG",
            "beta.jpg",
            "notes.txt",
            ".hidden.png",
            "gamma.webp",
        ] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join("Sub Dir")).unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        dir
    }

    #[test]
    fn matches_lists_dirs_then_images_and_skips_the_rest() {
        let dir = fixture();
        let prefix = format!("{}/", dir.path().display());
        assert_eq!(
            image_path_matches(&prefix),
            ["Sub Dir/", "Alpha.PNG", "beta.jpg", "gamma.webp"]
        );
        assert_eq!(image_path_matches(&format!("{prefix}A")), ["Alpha.PNG"]);
        assert_eq!(image_path_matches(&format!("{prefix}a")), ["Alpha.PNG"]);
        assert_eq!(
            image_path_matches(&format!("{prefix}.")),
            [".git/", ".hidden.png"]
        );
        assert!(image_path_matches(&format!("{prefix}zzz")).is_empty());
        assert!(image_path_matches("/definitely/not/a/dir/").is_empty());
    }

    #[test]
    fn path_argument_quotes_and_terminates_by_kind() {
        assert_eq!(path_argument("pi", "pics/"), "pics/");
        assert_eq!(path_argument("pics/sh", "shot.png"), "pics/shot.png ");
        assert_eq!(path_argument("", "Sub Dir/"), "\"Sub Dir/");
        assert_eq!(
            path_argument("Sub Dir/", "my shot.png"),
            "\"Sub Dir/my shot.png\" "
        );
        assert_eq!(path_argument("pics/", "sho"), "pics/sho");
    }

    #[test]
    fn common_prefix_is_case_insensitive() {
        let names = ["Shot1.png".to_string(), "shot2.png".to_string()];
        assert_eq!(common_prefix(&names), "Shot");
        assert_eq!(common_prefix(&["one.png".to_string()]), "one.png");
        assert_eq!(common_prefix(&[]), "");
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn candidates_lead_with_the_common_extension() {
        assert_eq!(
            candidates_for("s", &names(&["shot1.png", "shot2.png"])),
            ["shot", "shot1.png ", "shot2.png "]
        );
        // Already at the common prefix: no extension entry.
        assert_eq!(
            candidates_for("shot", &names(&["shot1.png", "shot2.png"])),
            ["shot1.png ", "shot2.png "]
        );
        assert_eq!(candidates_for("z", &names(&["zed.jpg"])), ["zed.jpg "]);
        assert_eq!(
            candidates_for("", &names(&["pics/", "zed.jpg"])),
            ["pics/", "zed.jpg "]
        );
        assert!(candidates_for("q", &[]).is_empty());
    }

    #[test]
    fn completions_start_at_the_argument_and_cover_the_quote() {
        let dir = fixture();
        let line = format!("/image {}/a", dir.path().display());
        let (start, candidates) = image_completions(&line, line.len()).unwrap();
        assert_eq!(start, "/image ".len());
        assert_eq!(candidates, [format!("{}/Alpha.PNG ", dir.path().display())]);

        let line = format!("/image \"{}/Sub", dir.path().display());
        let (start, candidates) = image_completions(&line, line.len()).unwrap();
        assert_eq!(start, "/image ".len());
        assert_eq!(candidates, [format!("\"{}/Sub Dir/", dir.path().display())]);

        assert_eq!(image_completions("/image a.png hi", 15), None);
        assert_eq!(image_completions("/model", 6), None);
        // Cursor not at the end: nothing to complete.
        assert_eq!(image_completions("/image ab", 8), None);
    }

    fn fake_dir(prefix: &str) -> Vec<String> {
        let (_, name) = split_dir_name(prefix);
        names(&["pics/", "shot1.png", "shot2.png", "zed.jpg"])
            .into_iter()
            .filter(|n| n.starts_with(name))
            .collect()
    }

    #[test]
    fn tab_completes_single_match_with_trailing_space() {
        let (step, cycle) = tab_step("/image z", true, None, fake_dir);
        assert_eq!(step, TabStep::Line("/image zed.jpg ".to_string()));
        // The next Tab shows the original again, then wraps.
        let (step, cycle) = tab_step("/image zed.jpg ", true, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image z".to_string()));
        let (step, _) = tab_step("/image z", true, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image zed.jpg ".to_string()));
    }

    #[test]
    fn tab_extends_to_common_prefix_then_cycles() {
        let (step, cycle) = tab_step("/image s", true, None, fake_dir);
        assert_eq!(step, TabStep::Line("/image shot".to_string()));
        let (step, cycle) = tab_step("/image shot", true, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image shot1.png ".to_string()));
        let (step, cycle) = tab_step("/image shot1.png ", true, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image shot2.png ".to_string()));
        let (step, cycle) = tab_step("/image shot2.png ", true, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image s".to_string()));
        let (step, _) = tab_step("/image s", false, cycle, fake_dir);
        assert_eq!(step, TabStep::Line("/image shot2.png ".to_string()));
    }

    #[test]
    fn backward_tab_starts_from_the_last_candidate() {
        let (step, _) = tab_step("/image ", false, None, fake_dir);
        assert_eq!(step, TabStep::Line("/image zed.jpg ".to_string()));
        let (step, _) = tab_step("/image ", true, None, fake_dir);
        assert_eq!(step, TabStep::Line("/image pics/".to_string()));
    }

    #[test]
    fn quoted_prefix_is_replaced_from_the_quote() {
        let dir = |_: &str| names(&["Sub Dir/"]);
        let (step, _) = tab_step("/image \"Su", true, None, dir);
        assert_eq!(step, TabStep::Line("/image \"Sub Dir/".to_string()));
    }

    #[test]
    fn typing_ends_the_cycle() {
        let (_, cycle) = tab_step("/image ", true, None, fake_dir);
        let (step, cycle) = tab_step("/image pics/x", true, cycle, |_| Vec::new());
        assert_eq!(step, TabStep::NoMatch);
        assert_eq!(cycle, None);
    }

    #[test]
    fn tab_passes_through_outside_the_path() {
        assert_eq!(
            tab_step("/image a.png hi", true, None, fake_dir).0,
            TabStep::NotImage
        );
        assert_eq!(
            tab_step("/model", true, None, fake_dir).0,
            TabStep::NotImage
        );
    }
}
