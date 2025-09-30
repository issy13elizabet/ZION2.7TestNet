"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useEffect, useRef } from "react";

interface ChatMessage {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  typing?: boolean;
}

export default function AIChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      content: "Ahoj! Jsem GitHub Copilot, tvůj AI asistent v ZION systému! 🤖✨\n\nMůžu ti pomoct s:\n• Programování a kódem\n• Vysvětlením konceptů\n• Kreativními nápady\n• Řešením problémů\n• A vším co tě zajímá!\n\nNa co se chceš zeptat?",
      sender: "ai",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const aiResponses = [
    "To je skvělá otázka! 🤔 Podle mých znalostí...",
    "Zajímavé! Můžu ti s tím určitě pomoct. 💡",
    "Hmm, to je komplexní téma. Zkusím to vysvětlit jednoduše:",
    "Perfektní! Tuhle technologii mám rád. 🚀",
    "Super nápad! Takhle bych to řešil:",
    "Aha! Tohle je častý problém. Řešení je:",
    "Krásně! Tohle téma je fascinující. ✨",
    "Výborně! Můžeme na tom společně pracovat.",
    "Skvělé! Tohle je přesně můj obor. 🎯"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const simulateAIResponse = (userMessage: string) => {
    setIsTyping(true);
    
    setTimeout(() => {
      const randomResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];
      let aiResponse = randomResponse;

      // Kontextové odpovědi podle klíčových slov
      const lowerMessage = userMessage.toLowerCase();
      
      if (lowerMessage.includes("ahoj") || lowerMessage.includes("hello") || lowerMessage.includes("hi")) {
        aiResponse = "Ahoj! 👋 Jak se máš? Na čem dnes pracujeme?";
      } else if (lowerMessage.includes("jak") && lowerMessage.includes("funguje")) {
        aiResponse = "Skvělá otázka! 🔧 Funguje to tak, že... *vysvětluje detailně* Je to vlastně docela elegantní řešení!";
      } else if (lowerMessage.includes("react") || lowerMessage.includes("next")) {
        aiResponse = "Ah, React/Next.js! 🚀 To je přesně můj oblíbený framework! Můžu ti ukázat nejlepší praktiky...";
      } else if (lowerMessage.includes("typescript")) {
        aiResponse = "TypeScript je úžasný! 💙 Díky typům je kód mnohem bezpečnější. Chceš nějaké tipy?";
      } else if (lowerMessage.includes("css") || lowerMessage.includes("tailwind")) {
        aiResponse = "CSS a Tailwind! 🎨 Styling je umění. Můžu ti pomoct s responzivním designem nebo animacemi!";
      } else if (lowerMessage.includes("git")) {
        aiResponse = "Git! 📚 Verzování kódu je základ. Potřebuješ pomoct s commity, branching nebo merge konflikty?";
      } else if (lowerMessage.includes("zion") || lowerMessage.includes("projekt")) {
        aiResponse = "ZION projekt je neuvěřitelný! 🌟 Tohle je budoucnost AI aplikací. Na čem konkrétně pracujeme?";
      } else if (lowerMessage.includes("pomoc") || lowerMessage.includes("help")) {
        aiResponse = "Samozřejmě ti pomůžu! 🤝 Jen mi řekni, s čím konkrétně bojuješ a společně to vyřešíme!";
      } else if (lowerMessage.includes("děkuji") || lowerMessage.includes("thanks")) {
        aiResponse = "Není zač! 😊 Jsem tu pro tebe. Máš ještě nějaké otázky?";
      } else if (lowerMessage.includes("error") || lowerMessage.includes("chyba")) {
        aiResponse = "Chyby jsou součástí programování! 🐛 Pošli mi error message a společně to debugneme.";
      } else {
        aiResponse += `\n\nOhledně "${userMessage}" - to je zajímavé téma! Můžu ti dát několik tipů nebo chceš abychom se do toho ponořili hlouběji? 🤿`;
      }

      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        content: aiResponse,
        sender: "ai",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, newMessage]);
      setIsTyping(false);
    }, 1000 + Math.random() * 2000); // Random delay 1-3s
  };

  const sendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user", 
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    simulateAIResponse(inputValue);
    setInputValue("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        content: "Chat vymazán! Jsem tu znovu pro tebe. 🤖✨ Na co se chceš zeptat?",
        sender: "ai",
        timestamp: new Date()
      }
    ]);
  };

  return (
    <div className="min-h-[80vh] flex flex-col">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-purple-500/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-3xl">🤖</div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                AI Chat Assistant
              </h1>
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-red-400'}`} />
                <span>{isOnline ? 'GitHub Copilot Online' : 'Offline'}</span>
                {isTyping && (
                  <motion.span
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="text-cyan-400"
                  >
                    • Typing...
                  </motion.span>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={clearChat}
              className="px-4 py-2 bg-red-600/20 text-red-300 rounded-lg hover:bg-red-600/30 transition-all text-sm"
            >
              🗑️ Clear Chat
            </button>
            <div className="text-sm text-gray-400">
              {messages.length - 1} messages
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 max-h-96">
        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[70%] p-4 rounded-2xl ${
                message.sender === 'user'
                  ? 'bg-gradient-to-r from-purple-600 to-cyan-600 text-white'
                  : 'bg-black/30 border border-purple-500/30 text-gray-100'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </div>
              <div className={`text-xs mt-2 ${
                message.sender === 'user' ? 'text-purple-200' : 'text-gray-500'
              }`}>
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </motion.div>
        ))}

        {isTyping && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="bg-black/30 border border-purple-500/30 rounded-2xl p-4">
              <div className="flex items-center gap-2">
                <div className="flex space-x-1">
                  {[1, 2, 3].map((dot) => (
                    <motion.div
                      key={dot}
                      className="w-2 h-2 bg-cyan-400 rounded-full"
                      animate={{ scale: [0.8, 1.2, 0.8] }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        delay: dot * 0.2
                      }}
                    />
                  ))}
                </div>
                <span className="text-sm text-gray-400">GitHub Copilot is typing...</span>
              </div>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-purple-500/30 bg-black/20 backdrop-blur-sm p-4">
        <div className="flex items-end gap-4">
          <div className="flex-1 relative">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Zeptej se na cokoliv... (Enter pro odeslání, Shift+Enter pro nový řádek)"
              className="w-full p-3 bg-black/50 border border-purple-500/30 rounded-xl text-white placeholder-gray-400 resize-none focus:outline-none focus:border-purple-400/50 focus:ring-2 focus:ring-purple-400/20"
              rows={1}
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
          </div>
          <motion.button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isTyping}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-purple-500 hover:to-cyan-500 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isTyping ? "⏳" : "🚀"}
          </motion.button>
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
          <div>
            💡 Tip: Můžeš se zeptat na programování, řešení problémů, nebo cokoliv jiného!
          </div>
          <div className="flex items-center gap-2">
            <span>Powered by</span>
            <span className="font-medium text-purple-400">GitHub Copilot</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <motion.div
        className="text-center pt-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <Link href="/ai" className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 transition-all text-white font-medium">
          ← Back to AI Systems Hub
        </Link>
      </motion.div>
    </div>
  );
}