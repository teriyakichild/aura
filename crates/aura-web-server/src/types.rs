use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::session_store::SessionStore;
use crate::streaming::ToolResultMode;

/// Tracks in-flight request count for graceful shutdown.
pub struct ActiveRequestTracker {
    count: AtomicUsize,
    drained: Notify,
}

impl Default for ActiveRequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveRequestTracker {
    pub fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            drained: Notify::new(),
        }
    }

    pub fn increment(&self) {
        self.count.fetch_add(1, Ordering::Release);
    }

    pub fn decrement(&self) {
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.drained.notify_waiters();
        }
    }

    /// Resolves when count reaches zero. If already zero, resolves immediately.
    pub async fn wait_for_drain(&self) {
        loop {
            // Register BEFORE checking count to close TOCTOU gap:
            // if decrement() fires between register and check, the stored
            // permit ensures notified().await returns immediately.
            let notified = self.drained.notified();
            if self.count.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }
}

/// RAII guard that increments the active request counter on creation and
/// decrements it on drop. Pair with one construction per logical request so
/// every code path (normal completion, early return, panic, consumer drop)
/// produces exactly one decrement.
pub struct ActiveRequestGuard {
    tracker: Arc<ActiveRequestTracker>,
}

impl ActiveRequestGuard {
    pub fn new(tracker: Arc<ActiveRequestTracker>) -> Self {
        tracker.increment();
        Self { tracker }
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.tracker.decrement();
    }
}

/// Application state
pub struct AppState {
    pub configs: Arc<Vec<aura_config::Config>>,
    pub tool_result_mode: ToolResultMode,
    /// Maximum length for tool results (0 = no truncation)
    pub tool_result_max_length: usize,
    pub streaming_buffer_size: usize,
    /// Enable Aura custom SSE events (aura.tool_requested, aura.tool_start, aura.tool_complete, etc.)
    pub aura_custom_events: bool,
    /// Enable reasoning event emission (only when aura_custom_events is true)
    pub aura_emit_reasoning: bool,
    /// Surface raw upstream provider errors to clients (see
    /// `AURA_DEBUG_PROVIDER_ERRORS`).
    pub debug_provider_errors: bool,
    /// SSE streaming request timeout in seconds (0 = no timeout)
    pub streaming_timeout_secs: u64,
    /// First chunk timeout in seconds (0 = disabled). Protects against hung provider connections.
    pub first_chunk_timeout_secs: u64,
    /// Inactivity timeout in seconds (0 = disabled).
    pub stream_inactivity_timeout_secs: u64,
    /// Shutdown gate — cancelled immediately on SIGTERM/SIGINT to reject new requests (503)
    pub shutdown_token: CancellationToken,
    /// Stream shutdown — cancelled after grace period to terminate in-flight streams
    pub stream_shutdown_token: CancellationToken,
    /// Tracks in-flight requests for early shutdown when all requests complete
    pub active_requests: Arc<ActiveRequestTracker>,
    /// Default agent name or alias, used when `model` is omitted from the request
    pub default_agent: Option<String>,
    /// Factory for additional tools to register on every agent (e.g., CLI tools in standalone mode).
    /// Called once per request to produce fresh tool instances. Returns empty vec for the web server.
    pub additional_tools: Arc<dyn Fn() -> Vec<Box<dyn aura::ToolDyn>> + Send + Sync>,
    pub pending_approvals: aura::hitl::PendingApprovals,
    /// Startup-loaded HMAC secret for the HITL webhook route (egress signing).
    pub hitl_webhook_hmac: Option<aura::hitl::WebhookHmac>,
    /// The session-state backend.
    pub session_store: Arc<dyn SessionStore>,
}

/// OpenAI-compatible message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool result message — used in client-side tool follow-ups, where the
    /// client executed a tool locally and is sending the result back.
    Tool,
    /// Catch-all for roles we don't handle (e.g. "function").
    #[serde(other, rename = "unknown")]
    Unknown,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
            Role::Unknown => write!(f, "unknown"),
        }
    }
}

/// OpenAI-compatible chat message structure.
///
/// `content` is optional because assistant messages with `tool_calls` may have
/// no text content, and `tool_call_id`/`name` are only populated for `Role::Tool`
/// follow-ups.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Tool calls emitted by an assistant message. Used when reconstructing
    /// conversation history for client-side tool follow-ups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    /// Identifier of the tool call this message is a result for (`role: "tool"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name (`role: "tool"`). Optional — present in OpenAI's spec but not
    /// required for correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content in either OpenAI wire shape: a plain string, or an array
/// of typed parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// The textual view of the content: the string itself, or the text parts
    /// joined with newlines (image parts contribute nothing).
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Text(text) => std::borrow::Cow::Borrowed(text),
            Self::Parts(parts) => std::borrow::Cow::Owned(
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.as_str()),
                        ContentPart::ImageUrl { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

/// One element of an array-shaped message `content` (OpenAI shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// An `image_url` content part: a `data:` URL carrying base64 image bytes.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Elides the payload of a `data:` URL so a `{:?}` of a request never dumps
/// base64 into logs or spans.
impl std::fmt::Debug for ImageUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let url: std::borrow::Cow<'_, str> = match self.url.split_once(',') {
            Some((head, payload)) if head.starts_with("data:") => {
                format!("{head},<{} bytes elided>", payload.len()).into()
            }
            _ => self.url.as_str().into(),
        };
        f.debug_struct("ImageUrl")
            .field("url", &url)
            .field("detail", &self.detail)
            .finish()
    }
}

/// Requested image fidelity (OpenAI `detail`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

impl From<ImageDetail> for aura::ImageDetail {
    fn from(detail: ImageDetail) -> Self {
        match detail {
            ImageDetail::Low => Self::Low,
            ImageDetail::High => Self::High,
            ImageDetail::Auto => Self::Auto,
        }
    }
}

/// A tool call emitted by an assistant message (OpenAI shape).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ChatMessageFunctionCall,
}

/// Function-call payload inside a `ChatMessageToolCall`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessageFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Client-side tool definition supplied via the OpenAI `tools` request field.
///
/// Converts to [`aura::builder::ClientTool`] via the `From` impl below.
#[derive(Debug, Deserialize, Clone)]
pub struct ClientToolDefinition {
    /// Always `"function"` in OpenAI's API today — accepted for spec fidelity.
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub tool_type: String,
    pub function: ClientFunctionDefinition,
}

/// Inner function definition of a client-side tool.
#[derive(Debug, Deserialize, Clone)]
pub struct ClientFunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

impl From<&ClientToolDefinition> for aura::builder::ClientTool {
    fn from(def: &ClientToolDefinition) -> Self {
        Self {
            name: def.function.name.clone(),
            description: def.function.description.clone().unwrap_or_default(),
            parameters: def
                .function
                .parameters
                .clone()
                .unwrap_or(serde_json::json!({})),
        }
    }
}

/// OpenAI-compatible chat completions request
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,

    /// OpenAI-compatible metadata field (up to 16 key-value pairs)
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,

    /// OpenAI-compatible end-user identifier.
    #[serde(default)]
    pub user: Option<String>,

    /// Client-side tool definitions. Per-agent opt-in via
    /// `[agent].enable_client_tools` and `[orchestration.worker.<name>].enable_client_tools`
    /// in TOML; tools are filtered with the matching `client_tool_filter`.
    #[serde(default)]
    pub tools: Option<Vec<ClientToolDefinition>>,
}

/// OpenAI-compatible error responses for chat completion requests.
pub enum ChatCompletionErrorResponse {
    /// Model parameter was not provided (HTTP 400).
    ModelNotProvided,
    /// Model was specified but does not match any configured agent (HTTP 404).
    ModelNotFound(String),
}

#[derive(Serialize)]
struct ChatCompletionErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: String,
    code: String,
}

#[derive(Serialize)]
struct ChatCompletionErrorEnvelope {
    error: ChatCompletionErrorDetail,
}

impl Serialize for ChatCompletionErrorResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (message, code) = match self {
            ChatCompletionErrorResponse::ModelNotProvided => (
                "you must provide a model parameter".to_string(),
                "missing_required_parameter".to_string(),
            ),
            ChatCompletionErrorResponse::ModelNotFound(model_name) => (
                format!(
                    "The model `{}` does not exist or you do not have access to it.",
                    model_name
                ),
                "model_not_found".to_string(),
            ),
        };

        let envelope = ChatCompletionErrorEnvelope {
            error: ChatCompletionErrorDetail {
                message,
                error_type: "invalid_request_error".to_string(),
                param: "model".to_string(),
                code,
            },
        };

        envelope.serialize(serializer)
    }
}

/// OpenAI-compatible choice structure
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Usage statistics
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// OpenAI-compatible chat completions response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,

    /// OpenAI-compatible metadata field (return session info)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Error response structure
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_active_request_tracker_immediate_drain() {
        let tracker = ActiveRequestTracker::new();
        // count=0 should resolve immediately
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            tracker.wait_for_drain(),
        )
        .await
        .expect("wait_for_drain should resolve immediately when count is 0");
    }

    #[tokio::test]
    async fn test_active_request_tracker_drain_after_decrement() {
        let tracker = Arc::new(ActiveRequestTracker::new());
        tracker.increment();

        let tracker_clone = tracker.clone();
        let handle = tokio::spawn(async move {
            tracker_clone.wait_for_drain().await;
        });

        // Give waiter time to register
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        tracker.decrement();

        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("wait_for_drain should resolve after decrement")
            .expect("task should not panic");
    }

    #[tokio::test]
    async fn test_active_request_tracker_multiple_requests() {
        let tracker = Arc::new(ActiveRequestTracker::new());
        tracker.increment();
        tracker.increment();
        tracker.increment();

        let tracker_clone = tracker.clone();
        let handle = tokio::spawn(async move {
            tracker_clone.wait_for_drain().await;
        });

        // Give waiter time to register
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Decrement one by one - should not resolve until count hits 0
        tracker.decrement();
        tracker.decrement();

        // Still at count=1, should not resolve yet
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!handle.is_finished(), "Should not resolve with count > 0");

        tracker.decrement(); // Now count=0

        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("wait_for_drain should resolve when count reaches 0")
            .expect("task should not panic");
    }

    #[test]
    fn test_model_not_provided_serialization() {
        let error = ChatCompletionErrorResponse::ModelNotProvided;
        let json: serde_json::Value = serde_json::to_value(&error).unwrap();

        assert_eq!(
            json["error"]["message"],
            "you must provide a model parameter"
        );
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "model");
        assert_eq!(json["error"]["code"], "missing_required_parameter");
    }

    #[test]
    fn test_model_not_found_serialization() {
        let error = ChatCompletionErrorResponse::ModelNotFound("gpt-5".to_string());
        let json: serde_json::Value = serde_json::to_value(&error).unwrap();

        assert_eq!(
            json["error"]["message"],
            "The model `gpt-5` does not exist or you do not have access to it."
        );
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["param"], "model");
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[test]
    fn message_content_accepts_string_and_parts() {
        let string: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user", "content": "hi"
        }))
        .unwrap();
        assert_eq!(string.content, Some(MessageContent::Text("hi".into())));

        let parts: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}
            ]
        }))
        .unwrap();
        let content = parts.content.unwrap();
        assert_eq!(content.text(), "what is this?");
        assert_eq!(
            content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".into()
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,AAAA".into(),
                        detail: Some(ImageDetail::Low),
                    }
                },
            ])
        );
    }

    #[test]
    fn message_content_rejects_unknown_part_types_and_details() {
        let audio = serde_json::from_value::<ChatMessage>(serde_json::json!({
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]
        }));
        assert!(audio.is_err());

        let detail = serde_json::from_value::<ChatMessage>(serde_json::json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "ultra"}}]
        }));
        assert!(detail.is_err());
    }

    #[test]
    fn string_content_serializes_as_a_plain_string() {
        let msg = ChatMessage {
            role: Role::User,
            content: Some("hi".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], serde_json::json!("hi"));
    }

    #[test]
    fn image_url_debug_elides_data_payload() {
        let image = ImageUrl {
            url: "data:image/png;base64,iVBORw0KGgo".into(),
            detail: None,
        };
        let debug = format!("{image:?}");
        assert!(!debug.contains("iVBOR"), "{debug}");
        assert!(
            debug.contains("data:image/png;base64,<11 bytes elided>"),
            "{debug}"
        );
    }
}
