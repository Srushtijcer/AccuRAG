import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import AuthPage from "./components/Auth/AuthPage";
import TopNavbar from "./components/TopNavbar";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import DocumentModal from "./components/DocumentModal";
import SettingsModal from "./components/SettingsModal";
import LightboxModal from "./components/LightboxModal";

import {
  getCurrentSession,
  logoutSession
} from "./services/auth";
import {
  checkBackendStatus,
  getDocuments,
  askQuery,
  uploadDocument
} from "./services/api";

export default function App() {
  // Authentication State
  const [user, setUser] = useState(() => getCurrentSession());

  // Theme State ("dark" | "light")
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("accurag_theme") || "dark";
  });

  // Navigation / Sidebar State
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    return localStorage.getItem("accurag_sidebar_collapsed") === "true";
  });
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Modals
  const [docModalOpen, setDocModalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const [lightboxUrl, setLightboxUrl] = useState(null);

  // Backend Health
  const [backendOnline, setBackendOnline] = useState(false);

  // Document Library & Active Doc
  const [indexedDocs, setIndexedDocs] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);

  // Chat State
  const [question, setQuestion] = useState("");
  const [answerMode, setAnswerMode] = useState("smart"); // 'smart' | 'document' | 'web'
  const [explanationMode, setExplanationMode] = useState("detailed"); // 'easy' | 'detailed' | 'exam'
  const [answerLength, setAnswerLength] = useState("medium"); // 'short' | 'medium' | 'detailed'
  const [examMarks, setExamMarks] = useState(null); // null | 2 | 5 | 10 | 15
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState("");
  const [currentResult, setCurrentResult] = useState(null);
  const [activeChatId, setActiveChatId] = useState(null);
  // Conversation session tracking for follow-up context
  const conversationIdRef = useRef(null);

  // Chat History
  const [chatHistory, setChatHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("accurag_chat_history");
      return saved
        ? JSON.parse(saved)
        : [
            {
              id: "hist-demo-1",
              question: "Explain the symmetric cipher model with a neat diagram.",
              timestamp: "Earlier",
              mode: "document",
            },
            {
              id: "hist-demo-2",
              question: "What are the latest developments in retrieval-augmented generation?",
              timestamp: "Earlier",
              mode: "web",
            },
          ];
    } catch {
      return [];
    }
  });

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("accurag_theme", theme);
  }, [theme]);

  // Persist sidebar collapsed state
  useEffect(() => {
    localStorage.setItem("accurag_sidebar_collapsed", sidebarCollapsed);
  }, [sidebarCollapsed]);

  // Persist chat history
  useEffect(() => {
    try {
      localStorage.setItem("accurag_chat_history", JSON.stringify(chatHistory));
    } catch (e) {
      console.error("Could not persist chat history:", e);
    }
  }, [chatHistory]);

  // Check backend health & fetch indexed documents on mount
  useEffect(() => {
    const initApp = async () => {
      try {
        const health = await checkBackendStatus();
        setBackendOnline(health.online);

        if (health.online) {
          const docData = await getDocuments();
          if (docData?.documents && docData.documents.length > 0) {
            setIndexedDocs(docData.documents);
            if (!activeDoc) {
              setActiveDoc(docData.documents[0]);
            }
          }
        }
      } catch (err) {
        console.warn("Backend initialization ping failed:", err);
        setBackendOnline(false);
      }
    };
    initApp();
  }, []);

  // Theme Toggle Handler
  const handleToggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Logout Handler
  const handleLogout = () => {
    logoutSession();
    setUser(null);
  };

  // New Chat Handler
  const handleNewChat = () => {
    setQuestion("");
    setCurrentResult(null);
    setAskError("");
    setActiveChatId(null);
    // Reset conversation session so next question starts fresh
    conversationIdRef.current = null;
  };

  // Clear History
  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear your chat history?")) {
      setChatHistory([]);
      localStorage.removeItem("accurag_chat_history");
    }
  };

  // Select History Item
  const handleSelectChat = (item) => {
    setActiveChatId(item.id);
    setQuestion(item.question);
    if (item.fullResult) {
      setCurrentResult(item.fullResult);
      if (item.fullResult.mode) setAnswerMode(item.fullResult.mode);
      if (item.fullResult.explanation_mode) setExplanationMode(item.fullResult.explanation_mode);
    } else {
      handleAsk(item.question, item.mode || answerMode);
    }
  };

  // Ask Question Handler
  const handleAsk = async (queryText = question, overrideMode = null, extraOptions = {}) => {
    const query = (queryText || "").trim();
    if (!query || asking) return;

    const modeToUse = overrideMode || answerMode;

    // Build previous context for follow-up awareness
    let previousContext = null;
    if (currentResult && currentResult.answer) {
      previousContext = {
        previous_question: currentResult.question || null,
        previous_answer: currentResult.answer || null,
        previous_sources: currentResult.sources || [],
        previous_web_sources: currentResult.web_sources || [],
        previous_diagrams: currentResult.diagrams || [],
        previous_mode: currentResult.mode || modeToUse,
        previous_page: currentResult.page || null,
        previous_format: currentResult.answer_format || null,
      };
    }

    try {
      setAsking(true);
      setAskError("");

      const res = await askQuery(query, modeToUse, explanationMode, {
        answerLength: extraOptions.answerLength || answerLength,
        examMarks: extraOptions.examMarks !== undefined ? extraOptions.examMarks : examMarks,
        conversationId: conversationIdRef.current,
        previousContext,
      });

      // Save the conversation_id from backend for session continuity
      if (res.conversation_id) {
        conversationIdRef.current = res.conversation_id;
      }

      setCurrentResult(res);

      const newHistoryItem = {
        id: `chat_${Date.now()}`,
        question: query,
        timestamp: "Just now",
        mode: modeToUse,
        fullResult: res,
      };

      setActiveChatId(newHistoryItem.id);
      setChatHistory((prev) => [newHistoryItem, ...prev.filter((p) => p.question !== query).slice(0, 24)]);

    } catch (err) {
      console.error("Ask query error:", err);
      setAskError(
        err.response?.data?.detail ||
          "Failed to retrieve verified response. Please verify that the backend is running at http://127.0.0.1:8000."
      );
    } finally {
      setAsking(false);
    }
  };

  // ================= VIEW RENDERING =================
  // If user is not authenticated, show Page 1 (AuthPage)
  if (!user) {
    return <AuthPage onLoginSuccess={(u) => setUser(u)} />;
  }

  // Otherwise, show Page 2 (Spacious AI Research Workspace)
  return (
    <div className={`accurag-app-shell ${theme}`}>
      {/* Compact Top Navigation Bar */}
      <TopNavbar
        user={user}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        onLogout={handleLogout}
        backendOnline={backendOnline}
        activeDoc={activeDoc}
        onOpenDocLibrary={() => setDocModalOpen(true)}
        onOpenSettings={() => setSettingsModalOpen(true)}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        mobileSidebarOpen={mobileSidebarOpen}
        onToggleMobileSidebar={() => setMobileSidebarOpen(!mobileSidebarOpen)}
      />

      {/* Main Layout Area: Sidebar + Main Workspace */}
      <div className="accurag-workspace-body">
        {/* Left Narrow Collapsible Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          mobileOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
          chatHistory={chatHistory}
          activeChatId={activeChatId}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          onClearHistory={handleClearHistory}
          onOpenDocLibrary={() => setDocModalOpen(true)}
          onOpenSettings={() => setSettingsModalOpen(true)}
          docCount={indexedDocs.length}
          user={user}
          onLogout={handleLogout}
        />

        {/* Central Research Workspace */}
        <main className="accurag-main-content">
          <ChatArea
            currentResult={currentResult}
            question={question}
            setQuestion={setQuestion}
            onAsk={handleAsk}
            asking={asking}
            askError={askError}
            answerMode={answerMode}
            setAnswerMode={setAnswerMode}
            explanationMode={explanationMode}
            setExplanationMode={setExplanationMode}
            answerLength={answerLength}
            setAnswerLength={setAnswerLength}
            examMarks={examMarks}
            setExamMarks={setExamMarks}
            onOpenDocUpload={() => setDocModalOpen(true)}
            onOpenDocLibrary={() => setDocModalOpen(true)}
            activeDoc={activeDoc}
            onOpenLightbox={(url) => setLightboxUrl(url)}
          />
        </main>
      </div>

      {/* Document Library Modal */}
      <DocumentModal
        isOpen={docModalOpen}
        onClose={() => setDocModalOpen(false)}
        indexedDocs={indexedDocs}
        setIndexedDocs={setIndexedDocs}
        activeDoc={activeDoc}
        setActiveDoc={setActiveDoc}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        backendOnline={backendOnline}
      />

      {/* Visual Evidence Lightbox Modal */}
      <LightboxModal
        imageUrl={lightboxUrl}
        onClose={() => setLightboxUrl(null)}
      />
    </div>
  );
}