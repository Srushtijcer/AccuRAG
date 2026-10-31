import axios from "axios";

export const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 120s timeout for heavy PDF embeddings and deep searches
});

/**
 * Check if the FastAPI backend is running
 */
export const checkBackendStatus = async () => {
  try {
    const response = await api.get("/", { timeout: 4000 });
    return { online: true, data: response.data };
  } catch (error) {
    return { online: false, error: error.message };
  }
};

/**
 * Get list of currently indexed documents in AccuRAG
 */
export const getDocuments = async () => {
  const response = await api.get("/documents");
  return response.data;
};

/**
 * Upload a PDF to the AccuRAG backend
 */
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

/**
 * Query the AccuRAG backend for answers using the POST endpoint
 * with full conversational context and parameter support.
 *
 * @param {string} question       - The user's question text
 * @param {string} mode           - 'smart', 'document', or 'web'
 * @param {string} explanationMode - 'easy', 'detailed', or 'exam'
 * @param {object} [options]      - Optional extended parameters
 * @param {string} [options.answerLength]    - 'short', 'medium', or 'detailed'
 * @param {number} [options.examMarks]       - 2, 5, 10, or 15
 * @param {string} [options.conversationId]  - Session identifier for follow-ups
 * @param {object} [options.previousContext] - Previous answer context for reformats
 */
export const askQuery = async (
  question,
  mode = "smart",
  explanationMode = "detailed",
  options = {}
) => {
  const {
    answerLength = "medium",
    examMarks = null,
    conversationId = null,
    previousContext = null,
  } = options;

  const payload = {
    question: question.trim(),
    mode,
    explanation_mode: explanationMode,
    answer_length: answerLength,
    exam_marks: examMarks || null,
    conversation_id: conversationId || null,
    previous_context: previousContext || null,
  };

  const response = await api.post("/ask", payload);
  return response.data;
};

/**
 * Standalone verification demo endpoint
 */
export const verifyClaim = async (text, sources = []) => {
  const response = await api.post("/verify", { text, sources });
  return response.data;
};

/**
 * Standalone query decomposition demo endpoint
 */
export const decomposeQuery = async (query) => {
  const response = await api.post("/decompose", { query });
  return response.data;
};

/**
 * Resolve full URL for diagram/source page image
 */
export const resolveDiagramUrl = (path) => {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
};

export default api;