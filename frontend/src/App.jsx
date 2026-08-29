import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [activePage, setActivePage] = useState("home");
  const [uploading, setUploading] = useState(false);

  const fileInputRef = useRef(null);

  // Open PDF picker
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // Handle PDF selection
 const handleFileChange = async (event) => {
  const file = event.target.files[0];

  if (!file) return;

  if (file.type !== "application/pdf") {
    alert("Please select a PDF file.");
    event.target.value = "";
    return;
  }

  try {
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Upload failed");
    }

    const data = await response.json();

    console.log("Upload response:", data);

    setSelectedFile(file);

    alert(
      `PDF uploaded successfully!\n\nPages: ${data.pages}\nText extracted successfully.`
    );
  } catch (error) {
    console.error("Upload error:", error);
    alert("PDF upload failed. Please check if the backend is running.");
  } finally {
    setUploading(false);
    event.target.value = "";
  }
};
  // New chat
  const handleNewChat = () => {
    setQuestion("");
    setSelectedFile(null);
    setActivePage("home");
  };

  // Ask question
  const handleAsk = () => {
    if (!question.trim()) {
      return;
    }

    alert(`Question: ${question}`);
  };

  return (
    <div className="app">

      {/* ================= SIDEBAR ================= */}

      <aside className="sidebar">

        {/* Logo */}

        <div className="logo">

          <div className="logo-icon">
            ✦
          </div>

          <div className="logo-text">
            Accu<span>RAG</span>
          </div>

        </div>


        {/* New Chat */}

        <button
          className="new-chat"
          onClick={handleNewChat}
        >
          ＋ &nbsp; New Chat
        </button>


        {/* Navigation */}

        <nav className="nav">

          {/* Search */}

          <div
            className="nav-item"
            onClick={() => setActivePage("search")}
          >
            <span className="nav-icon">
              ⌕
            </span>

            <span>
              Search
            </span>
          </div>


          {/* Documents */}

          <div
            className="nav-item"
            onClick={() => setActivePage("documents")}
          >
            <span className="nav-icon">
              ▣
            </span>

            <span>
              Documents
            </span>
          </div>


          {/* Upload PDF */}

          <div
            className="nav-item"
            onClick={handleUploadClick}
          >
            <span className="nav-icon">
              ↑
            </span>

            <span>
              Upload PDF
            </span>
          </div>


          {/* Sources */}

          <div
            className="nav-item"
            onClick={() => setActivePage("sources")}
          >
            <span className="nav-icon">
              ◇
            </span>

            <span>
              Sources
            </span>
          </div>


          {/* Chat History */}

          <div
            className="nav-item"
            onClick={() => setActivePage("history")}
          >
            <span className="nav-icon">
              ◷
            </span>

            <span>
              Chat History
            </span>
          </div>

        </nav>


        {/* Hidden PDF input */}

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />


        {/* Chat History */}

        <div className="history">

          <div className="history-title">
            Today
          </div>

          <div className="history-item">
            What is blockchain?
          </div>

          <div className="history-item">
            Explain consensus...
          </div>

          <div className="history-item">
            What is Hyperledger?
          </div>

          <div className="history-day">
            Yesterday
          </div>

          <div className="history-item">
            Explain smart contracts...
          </div>

          <div className="history-item">
            What is a distributed ledger?
          </div>

        </div>

      </aside>


      {/* ================= MAIN ================= */}

      <main className="main">

        {/* =====================================
            DOCUMENTS PAGE
        ====================================== */}

        {activePage === "documents" && (

          <div className="documents-page">

            <div className="documents-header">

              <div>

                <h1>
                  Your Documents
                </h1>

                <p>
                  PDFs available in your AccuRAG workspace
                </p>

              </div>


              <button
                className="document-upload-button"
                onClick={handleUploadClick}
              >
                + Upload PDF
              </button>

            </div>


            {selectedFile ? (

              <div className="document-list">

                <div className="document-item">

                  <div className="document-icon">
                    PDF
                  </div>


                  <div className="document-info">

                    <h3>
                      {selectedFile.name}
                    </h3>

                    <p>
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>

                  </div>


                  <div className="document-status">
                    Ready
                  </div>

                </div>

              </div>

            ) : (

              <div className="empty-documents">

                <div className="empty-icon">
                  ▣
                </div>

                <h2>
                  No documents yet
                </h2>

                <p>
                  Upload a PDF to start asking questions about it.
                </p>

                <button
  className="document-upload-button"
  onClick={handleUploadClick}
  disabled={uploading}
>
  {uploading ? "Uploading..." : "+ Upload PDF"}
</button>

              </div>

            )}

          </div>

        )}


        {/* =====================================
            SEARCH PAGE
        ====================================== */}

        {activePage === "search" && (

          <div className="documents-page">

            <div className="documents-header">

              <div>

                <h1>
                  Search
                </h1>

                <p>
                  Search information across your documents.
                </p>

              </div>

            </div>


            <div className="chat-box">

              <textarea
                className="chat-input"
                placeholder="Search your documents..."
                value={question}
                onChange={(e) =>
                  setQuestion(e.target.value)
                }
              />

              <div className="chat-bottom">

                <div className="chat-left">
                  <span className="mode">
                    Document Search
                  </span>
                </div>

                <button
                  className="send-button"
                  onClick={handleAsk}
                >
                  ↑
                </button>

              </div>

            </div>

          </div>

        )}


        {/* =====================================
            SOURCES PAGE
        ====================================== */}

        {activePage === "sources" && (

          <div className="documents-page">

            <div className="documents-header">

              <div>

                <h1>
                  Sources
                </h1>

                <p>
                  Sources used to generate AccuRAG answers.
                </p>

              </div>

            </div>


            <div className="empty-documents">

              <div className="empty-icon">
                ◇
              </div>

              <h2>
                No sources yet
              </h2>

              <p>
                Sources will appear here after asking questions.
              </p>

            </div>

          </div>

        )}


        {/* =====================================
            CHAT HISTORY PAGE
        ====================================== */}

        {activePage === "history" && (

          <div className="documents-page">

            <div className="documents-header">

              <div>

                <h1>
                  Chat History
                </h1>

                <p>
                  Your previous conversations.
                </p>

              </div>

            </div>


            <div className="document-list">

              <div className="document-item">

                <div className="document-icon">
                  Q
                </div>

                <div className="document-info">

                  <h3>
                    What is blockchain?
                  </h3>

                  <p>
                    Today
                  </p>

                </div>

              </div>


              <div className="document-item">

                <div className="document-icon">
                  Q
                </div>

                <div className="document-info">

                  <h3>
                    Explain consensus
                  </h3>

                  <p>
                    Today
                  </p>

                </div>

              </div>

            </div>

          </div>

        )}


        {/* =====================================
            HOME PAGE
        ====================================== */}

        {activePage === "home" && (

          <>

            {/* Top bar */}

            <div className="topbar">

              <button className="top-button">
                ⌘
              </button>

              <button className="top-button">
                ⋮
              </button>

              <div className="profile">
                S
              </div>

            </div>


            {/* Welcome */}

            <section className="welcome">

              <div className="ai-orb"></div>

             <h1>
  Welcome to <span>AccuRAG</span>
</h1>

<p>
  Not Just Answers. Verified Intelligence.
</p>

            </section>


            {/* Feature cards */}

            <section className="features">


              <div className="feature-card">

                <div className="feature-icon">
                  ◇
                </div>

                <h3>
                  Document Analysis
                </h3>

                <p>
                  Understand your PDFs and extract meaningful information.
                </p>

              </div>


              <div className="feature-card">

                <div className="feature-icon">
                  ✦
                </div>

                <h3>
                  Accurate Answers
                </h3>

                <p>
                  Get answers grounded in the content of your documents.
                </p>

              </div>


              <div className="feature-card">

                <div className="feature-icon">
                  ▣
                </div>

                <h3>
                  Smart Sources
                </h3>

                <p>
                  See the document pages used to generate each answer.
                </p>

              </div>


              <div className="feature-card">

                <div className="feature-icon">
                  ⌕
                </div>

                <h3>
                  Intelligent Search
                </h3>

                <p>
                  Find relevant information quickly across your documents.
                </p>

              </div>


            </section>


            {/* Selected PDF */}

            {selectedFile && (

              <div
                style={{
                  maxWidth: "1000px",
                  margin: "0 auto 20px",
                  padding: "15px",
                  border: "1px solid #303036",
                  borderRadius: "14px",
                  background: "#101012",
                  display: "flex",
                  alignItems: "center",
                  gap: "15px"
                }}
              >

                <div
                  style={{
                    color: "#a78bfa",
                    fontWeight: "bold"
                  }}
                >
                  PDF
                </div>


                <div style={{ flex: 1 }}>

                  <div>
                    {selectedFile.name}
                  </div>

                  <small style={{ color: "#777" }}>
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </small>

                </div>


                <button
                  onClick={() => setSelectedFile(null)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "#aaa",
                    fontSize: "20px",
                    cursor: "pointer"
                  }}
                >
                  ×
                </button>

              </div>

            )}


            {/* Chat */}

            <section className="chat-box">

              <textarea
                className="chat-input"
                value={question}
                onChange={(e) =>
                  setQuestion(e.target.value)
                }
                onKeyDown={(e) => {

                  if (
                    e.key === "Enter" &&
                    !e.shiftKey
                  ) {

                    e.preventDefault();

                    handleAsk();

                  }

                }}
                placeholder={
                  selectedFile
                    ? "Ask anything about your PDF..."
                    : "Ask anything about your documents..."
                }
              />


              <div className="chat-bottom">

                <div className="chat-left">

                  <button
                    className="add-button"
                    onClick={handleUploadClick}
                  >
                    +
                  </button>

                  <span className="mode">
                    Smart⌄
                  </span>

                </div>


                <button
                  className="send-button"
                  onClick={handleAsk}
                >
                  ↑
                </button>

              </div>

            </section>

          </>

        )}

      </main>

    </div>
  );
}

export default App;