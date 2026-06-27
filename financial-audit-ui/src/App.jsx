import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; 

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0); 
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false); 
  const [attachedFiles, setAttachedFiles] = useState([]); // mảng nhiều file - cho phép gửi đồng thời nhiều cấu phần BCTC (B01-DN, B02-DN...)

  const [chatSessions, setChatSessions] = useState([
    { id: 1, title: 'Rà soát rủi ro tài chính', messages: [], mongoSessionId: null }
  ]);
  const [currentSessionId, setCurrentSessionId] = useState(1);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const menuRef = useRef(null);

  const currentSession = chatSessions.find(s => s.id === currentSessionId) || chatSessions[0];
  const messages = currentSession.messages;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleCreateNewChat = async () => {
    const newId = Date.now();

    // Tạo phiên bộ nhớ phân tầng trên MongoDB ngay khi mở chat mới
    let mongoSessionId = null;
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/session/create', { method: 'POST' });
      const data = await res.json();
      mongoSessionId = data.session_id;
    } catch (e) {
      console.warn('Không tạo được session MongoDB, fallback về history array:', e);
    }

    const newSession = { id: newId, title: 'Cuộc trò chuyện mới', messages: [], mongoSessionId };
    setChatSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newId);
    setAttachedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    if (!newFiles.length) return;

    const allowedExtensions = /(\.pdf|\.docx|\.xlsx|\.xls)$/i;
    const invalidFile = newFiles.find(f => !allowedExtensions.exec(f.name));
    if (invalidFile) {
      alert("Hệ thống từ chối! Định dạng tệp không được hỗ trợ. Vui lòng chỉ đính kèm tệp PDF, Word (docx) hoặc Excel (xlsx, xls).");
      e.target.value = ''; 
      return;
    }

    // Cộng dồn vào danh sách hiện có - cho phép chọn nhiều lần để gửi
    // đồng thời nhiều cấu phần báo cáo (ví dụ B01-DN, B02-DN, B03-DN, B09-DN)
    setAttachedFiles(prev => [...prev, ...newFiles]);
    setMenuOpen(false);
    e.target.value = ''; // reset input để có thể chọn lại cùng tên file nếu cần
  };

  const handleRemoveFile = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if ((!input.trim() && attachedFiles.length === 0) || loading) return;

    const userQuery = input;
    const filesToSend = attachedFiles;
    const activeSessionId = currentSessionId;
    const mongoSessionId = currentSession.mongoSessionId; // ← lấy session MongoDB của phiên hiện tại

    // Lấy ra toàn bộ lịch sử trò chuyện TRƯỚC ĐÓ của đúng phiên này
    const currentHistory = [...messages];

    setInput('');
    setAttachedFiles([]); 
    if (fileInputRef.current) fileInputRef.current.value = ''; 
    
    setLoading(true);
    setStep(1); 

    let displayDisplayText = userQuery;
    if (filesToSend.length > 0) {
      const fileNamesLabel = filesToSend.map(f => f.name).join(', ');
      displayDisplayText = `[📎 Tệp đính kèm: ${fileNamesLabel}]` + (userQuery ? `\n${userQuery}` : '');
    }

    // Đẩy tin nhắn mới của người dùng vào giao diện trước
    setChatSessions(prevSessions => prevSessions.map(session => {
      if (session.id === activeSessionId) {
        const isFirstMessage = session.messages.length === 0;
        const newTitle = isFirstMessage 
          ? (filesToSend.length > 0 ? `Tệp: ${filesToSend[0].name}${filesToSend.length > 1 ? ` +${filesToSend.length - 1}` : ''}` : userQuery.substring(0, 18) + '...') 
          : session.title;
        
        return {
          ...session,
          title: newTitle,
          messages: [...session.messages, { sender: 'user', text: displayDisplayText }]
        };
      }
      return session;
    }));

    // ĐÓNG GÓI LỊCH SỬ: Gom cả tin nhắn mới chuẩn bị gửi đi vào mảng lịch sử hội thoại
    const fullHistoryPayload = [
      ...currentHistory,
      { sender: 'user', text: displayDisplayText }
    ];

    try {
      let response;
      
      if (filesToSend.length > 0) {
        const formData = new FormData();
        // Đính kèm nhiều file cùng field "files" - cho phép gửi đồng thời
        // nhiều cấu phần BCTC (B01-DN, B02-DN, B03-DN, B09-DN...) để Gemini
        // đối chiếu chéo ngay trong một lần xử lý, thay vì phải hỏi nhiều lượt.
        filesToSend.forEach(f => formData.append('files', f));
        formData.append('query', userQuery);
        formData.append('history', JSON.stringify(fullHistoryPayload));
        if (mongoSessionId) formData.append('session_id', mongoSessionId); // ← thêm session_id

        response = await fetch('http://127.0.0.1:8000/api/v1/review-file', {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch('http://127.0.0.1:8000/api/v1/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            query: userQuery,
            text: userQuery,
            prompt: userQuery,
            history: fullHistoryPayload,
            session_id: mongoSessionId || null, // ← thêm session_id
          }),
        });
      }

      setStep(2); 
      const data = await response.json();
      setStep(3); 

      if (!response.ok || data.detail) {
        let errorMsg = data.detail || "Đã xảy ra sự cố xử lý dữ liệu.";
        if (typeof errorMsg === 'object') errorMsg = JSON.stringify(errorMsg);
        
        if (errorMsg.includes("429") || errorMsg.includes("RESOURCE_EXHAUSTED")) {
          errorMsg = "⚠️ **Hệ thống quá tải tài nguyên (Quota Exceeded):** Vui lòng đợi 30 giây rồi thử lại câu hỏi ngắn hơn.";
        }
        throw new Error(errorMsg);
      }

      let aiResponseText = "Không nhận được nội dung phân tích từ máy chủ.";
      if (data) {
        aiResponseText = data.analysis || data.response || data.result || data.output || (typeof data === 'string' ? data : JSON.stringify(data));
      }
      const chunks = data.extracted_chunks_count !== undefined ? data.extracted_chunks_count : 0;
      
      setChatSessions(prevSessions => prevSessions.map(session => {
        if (session.id === activeSessionId) {
          return {
            ...session,
            messages: [...session.messages, { 
              sender: 'ai', 
              text: aiResponseText,
              chunksCount: chunks
            }]
          };
        }
        return session;
      }));

    } catch (error) {
      console.error("Lỗi kết nối API:", error);
      setChatSessions(prevSessions => prevSessions.map(session => {
        if (session.id === activeSessionId) {
          return {
            ...session,
            messages: [...session.messages, { 
              sender: 'ai', 
              text: error.message.startsWith("⚠️") ? error.message : `❌ **Lỗi máy chủ:** ${error.message || "Không thể kết nối hoặc xử lý dữ liệu từ Backend."}` 
            }]
          };
        }
        return session;
      }));
    } finally {
      setLoading(false);
      setStep(0);
    }
  };

  return (
    <div className="h-screen w-screen bg-[#0e0e11] text-[#e3e3e3] flex font-sans overflow-hidden fixed inset-0">
      
      {/* 1. THANH SIDEBAR BÊN TRÁI */}
      <div className={`${sidebarOpen ? 'w-68' : 'w-0'} bg-[#17171c] h-full flex flex-col justify-between transition-all duration-300 shrink-0 overflow-hidden border-r border-slate-800/40`}>
        <div className="flex flex-col flex-1 p-4 overflow-hidden">
          <div className="flex items-center mb-4 shrink-0">
            <button onClick={() => setSidebarOpen(false)} className="p-2 hover:bg-[#282a36] rounded-full transition text-slate-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" /></svg>
            </button>
          </div>

          <button onClick={handleCreateNewChat} className="w-full bg-[#212129] hover:bg-[#2c2c38] text-slate-200 text-xs font-medium py-3 px-4 rounded-full flex items-center gap-3 mb-4 transition border border-slate-800 shrink-0 shadow-sm">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-4 h-4 text-slate-400"><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
            <span>Cuộc trò chuyện mới</span>
          </button>

          <div className="text-xs font-medium text-slate-500 px-2 mb-2 shrink-0">Gần đây</div>
          <div className="flex-1 overflow-y-auto space-y-1 pr-1 select-none">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => setCurrentSessionId(session.id)}
                className={`p-2.5 rounded-xl flex items-center gap-2 text-xs cursor-pointer truncate transition ${session.id === currentSessionId ? 'bg-[#212129] text-teal-400 font-medium' : 'text-slate-400 hover:bg-[#1f1f26]'}`}
              >
                <span>{session.id === currentSessionId ? '✨' : '💬'}</span>
                <span className="truncate flex-1 text-left">{session.title}</span>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* 2. KHU VỰC KHÔNG GIAN CHAT CHÍNH */}
      <div className="flex-1 h-full flex flex-col relative bg-[#0e0e11] overflow-hidden">
        
        <header className="p-4 flex items-center gap-3 shrink-0">
          {!sidebarOpen && (
            <button onClick={() => setSidebarOpen(true)} className="p-2 hover:bg-[#212129] rounded-full transition text-slate-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" /></svg>
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto px-6 md:px-16 py-6 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col justify-center items-center text-center">
              <h1 className="text-4xl md:text-5xl font-medium tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570]">
                Bình tĩnh, tự tin và viết đúng chính tả
              </h1>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-8 pb-32">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex w-full ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}>
                  <div className={`text-sm leading-relaxed ${msg.sender === 'user' ? 'max-w-[70%] text-white text-right font-medium bg-[#212129] px-4 py-2.5 rounded-2xl rounded-tr-none border border-slate-800/60 whitespace-pre-wrap' : 'w-full text-left text-slate-200'}`}>
                    
                    {msg.sender === 'ai' ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          ul: ({node, ...props}) => <ul className="list-disc pl-5 my-2 space-y-1.5 text-[#e3e3e3]" {...props} />,
                          ol: ({node, ...props}) => <ol className="list-decimal pl-5 my-2 space-y-1.5 text-[#e3e3e3]" {...props} />,
                          li: ({node, ...props}) => <li className="leading-relaxed" {...props} />,
                          p: ({node, ...props}) => <p className="my-2.5 leading-relaxed text-[#e3e3e3]" {...props} />,
                          strong: ({node, ...props}) => <strong className="font-bold text-white tracking-wide" {...props} />,
                          code: ({node, ...props}) => <code className="bg-[#1e1f20] px-1.5 py-0.5 rounded text-teal-400 font-mono text-xs border border-slate-800" {...props} />,
                          table: ({node, ...props}) => (
                            <div className="overflow-x-auto my-3">
                              <table className="min-w-full text-xs border border-slate-700 rounded-lg" {...props} />
                            </div>
                          ),
                          thead: ({node, ...props}) => <thead className="bg-[#1e1f20]" {...props} />,
                          th: ({node, ...props}) => <th className="px-3 py-2 text-left text-teal-400 font-semibold border-b border-slate-700" {...props} />,
                          td: ({node, ...props}) => <td className="px-3 py-2 text-slate-300 border-b border-slate-800" {...props} />,
                          tr: ({node, ...props}) => <tr className="hover:bg-[#1a1a20] transition" {...props} />,
                        }}
                        
                      >
                        {msg.text}
                      </ReactMarkdown>
                    ) : (
                      <div>{msg.text}</div>
                    )}

                    {msg.sender === 'ai' && msg.chunksCount !== undefined && msg.chunksCount > 0 && (
                      <div className="mt-3 text-[11px] text-slate-500 font-mono tracking-wide select-none">
                        ✓ Căn cứ: Đã tìm thấy {msg.chunksCount} phân đoạn từ kho dữ liệu Thông tư 99.
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex w-full justify-start">
                  <div className="text-sm text-teal-400/80 animate-pulse font-medium font-mono">
                    {step === 1 && "✦ Đang tải và phân tích cấu trúc tệp dữ liệu..."}
                    {step === 2 && "✦ Đang bốc tách văn bản luật từ Atlas..."}
                    {step === 3 && "✦ Gemini đang đối chiếu lập luận..."}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Ô INPUT CHUẨN VIÊN NHỘNG */}
        <div className="p-4 bg-gradient-to-t from-[#0e0e11] via-[#0e0e11]/95 to-transparent shrink-0">
          <div className="max-w-3xl mx-auto relative flex flex-col items-center">
            
            {attachedFiles.length > 0 && (
              <div className="w-full max-w-2xl bg-[#1e1f20] border border-slate-800 rounded-t-xl divide-y divide-slate-800 animate-slideUp">
                {attachedFiles.map((f, idx) => (
                  <div key={idx} className="text-xs text-teal-400 px-4 py-2 flex justify-between items-center">
                    <span className="truncate">📎 Sẵn sàng gửi: <strong className="text-slate-200">{f.name}</strong> ({Math.round(f.size/1024)} KB)</span>
                    <button type="button" onClick={() => handleRemoveFile(idx)} className="text-rose-400 hover:text-rose-500 font-bold ml-2">Xóa</button>
                  </div>
                ))}
              </div>
            )}

            <form onSubmit={handleSendMessage} className="w-full relative flex items-center">
              <div className="absolute left-3 z-20" ref={menuRef}>
                <button
                  type="button"
                  onClick={() => setMenuOpen(!menuOpen)}
                  className="p-1.5 bg-[#2b2c35] hover:bg-[#373945] text-slate-300 rounded-full transition flex items-center justify-center shadow-md"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                </button>

                {menuOpen && (
                  <div className="absolute bottom-10 left-0 bg-[#1e1f20] border border-slate-800 rounded-2xl p-2 w-48 shadow-2xl animate-fadeIn z-30">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current.click()}
                      className="w-full text-left text-xs text-slate-300 hover:bg-[#2b2c35] py-2.5 px-3 rounded-xl transition flex items-center gap-2.5"
                    >
                      <svg xmlns="http://www.w3.org/2000/xl" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4 text-teal-400"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" /></svg>
                      <span>Tải tệp lên</span>
                    </button>
                    <div className="text-[10px] text-slate-500 px-3 pt-1 border-t border-slate-800 mt-1">Hỗ trợ: PDF, DOCX, XLSX</div>
                  </div>
                )}
              </div>

              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept=".pdf,.docx,.xlsx,.xls" 
                multiple
                className="hidden" 
              />

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={loading}
                placeholder="Hỏi hoặc đính kèm tệp rà soát báo cáo tài chính..."
                className="w-full bg-[#1e1f20] border border-transparent rounded-full pl-14 pr-14 py-3.5 text-sm focus:outline-none focus:bg-[#17171c] text-slate-200 placeholder-[#757575] transition-all"
              />
              
              <button
                type="submit"
                disabled={loading || (!input.trim() && attachedFiles.length === 0)}
                className="absolute right-3 p-2 bg-transparent hover:bg-[#212129] text-slate-400 hover:text-slate-200 disabled:text-slate-700 transition"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
                </svg>
              </button>
            </form>
            <p className="text-[10px] text-slate-600 mt-2">Hệ thống bốc tách tri thức tự động dựa trên tài liệu kế toán đầu vào.</p>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;