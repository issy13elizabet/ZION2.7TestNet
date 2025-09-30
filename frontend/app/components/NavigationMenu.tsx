"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

const NavigationMenu = () => {
  const pathname = usePathname();

  const menuItems = [
    {
      href: "/hub",
      label: "Home",
      icon: "🏠",
      description: "ZION Main Portal"
    },
    {
      href: "/explorer",
      label: "Explorer",
      icon: "🌐",
      description: "Blockchain Explorer"
    }
  ];

  return (
    <motion.nav 
      className="fixed top-6 left-6 right-6 z-[100] bg-black/20 backdrop-blur-md border border-purple-500/30 rounded-3xl p-4 overflow-visible"
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="flex items-center justify-between">
        {/* Logo */}
        <motion.div 
          className="flex items-center gap-3"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">Z</span>
          </div>
          <span className="text-white font-bold text-lg">ZION v2.5</span>
          <span className="text-purple-400 text-sm bg-purple-900/30 px-2 py-1 rounded">TestNet</span>
        </motion.div>

        {/* Navigation Items */}
        <div className="flex items-center gap-2">
          {menuItems.map((item, index) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            
            return (
              <motion.div
                key={item.href}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={item.href}>
                  <motion.div
                    className={`
                      relative px-4 py-2 rounded-xl transition-all duration-300 group cursor-pointer
                      ${isActive 
                        ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg' 
                        : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
                      }
                    `}
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{item.icon}</span>
                      <span className="font-medium hidden md:inline">{item.label}</span>
                    </div>
                    
                    {/* Tooltip */}
                    <motion.div
                      className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-black/90 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap pointer-events-none"
                      initial={{ opacity: 0, scale: 0.8 }}
                      whileHover={{ opacity: 1, scale: 1 }}
                    >
                      {item.description}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-black/90"></div>
                    </motion.div>
                  </motion.div>
                </Link>
              </motion.div>
            );
          })}

          {/* AI Systems Dropdown */
          }
          <AISystemsDropdown currentPath={pathname} />

          {/* Temples Dropdown */}
          <TemplesDropdown currentPath={pathname} />
        </div>

        {/* Navigation Items */}
        <div className="flex items-center gap-2">
          {/* Dashboard Dropdown */}
          <DashboardDropdown currentPath={pathname} />
          {/* Network Status */}
          <motion.div 
            className="flex items-center gap-2 text-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <motion.div 
              className="w-2 h-2 bg-green-400 rounded-full"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-green-400 hidden lg:inline">Online</span>
          </motion.div>

          {/* Connection Quality */}
          <motion.div 
            className="flex items-center gap-1"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            {[1, 2, 3].map((bar) => (
              <motion.div
                key={bar}
                className={`w-1 bg-blue-400 rounded-full ${
                  bar === 1 ? 'h-2' : bar === 2 ? 'h-3' : 'h-4'
                }`}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ 
                  duration: 1.5, 
                  repeat: Infinity, 
                  delay: bar * 0.2 
                }}
              />
            ))}
          </motion.div>
        </div>
      </div>
    </motion.nav>
  );
};

export default NavigationMenu;

// Sub-component: AI Systems dropdown grouping AI home and OASIS GAME
function AISystemsDropdown({ currentPath }: { currentPath: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);
  const isActive = ["/ai", "/oasis-game", "/zion-os", "/ai-chat", "/zion-rig", "/zion-craft"].some(p => currentPath === p || currentPath.startsWith(p + "/"));

  return (
    <div
      ref={ref}
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <motion.button
        className={`relative px-4 py-2 rounded-xl transition-all duration-300 group cursor-pointer flex items-center gap-2 ${
          isActive ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg' : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
        }`}
        whileHover={{ scale: 1.05, y: -2 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(v => !v)}
      >
        <span className="text-lg">🤖</span>
        <span className="font-medium hidden md:inline">AI Systems</span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="hidden md:inline"
        >
          ▼
        </motion.span>

        {/* Tooltip */}
        <motion.div
          className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-black/90 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap pointer-events-none"
          initial={{ opacity: 0, scale: 0.8 }}
          whileHover={{ opacity: 1, scale: 1 }}
        >
          AI Module Control
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-black/90" />
        </motion.div>
      </motion.button>

      {/* Dropdown menu */}
      <motion.div
        initial={false}
        animate={{ opacity: open ? 1 : 0, y: open ? 0 : -8, pointerEvents: open ? 'auto' : 'none' }}
        transition={{ duration: 0.15 }}
        className="absolute right-0 mt-2 w-64 bg-black/80 backdrop-blur-md border border-purple-500/30 rounded-2xl overflow-hidden shadow-xl z-[300]"
      >
        <Link href="/ai" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/ai') ? 'text-white' : 'text-gray-300'}`}>
            <span>🤖</span>
            <div>
              <div className="font-medium">AI Systems Home</div>
              <div className="text-xs text-gray-400">Overview of AI modules</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/oasis-game" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/oasis-game') ? 'text-white' : 'text-gray-300'}`}>
            <span>🎮</span>
            <div>
              <div className="font-medium">OASIS GAME</div>
              <div className="text-xs text-gray-400">Playful path to mastery</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/zion-os" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/zion-os') ? 'text-white' : 'text-gray-300'}`}>
            <span>💻</span>
            <div>
              <div className="font-medium">ZION OS</div>
              <div className="text-xs text-gray-400">AI-Powered Operating System</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/ai-chat" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/ai-chat') ? 'text-white' : 'text-gray-300'}`}>
            <span>🗣️</span>
            <div>
              <div className="font-medium">AI Chat</div>
              <div className="text-xs text-gray-400">Talk with GitHub Copilot</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/zion-rig" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/zion-rig') ? 'text-white' : 'text-gray-300'}`}>
            <span>⚡</span>
            <div>
              <div className="font-medium">ZION RIG</div>
              <div className="text-xs text-gray-400">Hybrid System Control</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/zion-craft" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/zion-craft') ? 'text-white' : 'text-gray-300'}`}>
            <span>🧱</span>
            <div>
              <div className="font-medium">ZION CRAFT</div>
              <div className="text-xs text-gray-400">Minecraft × Roblox Fusion</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/stargate" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/stargate') ? 'text-white' : 'text-gray-300'}`}>
            <span>🌀</span>
            <div>
              <div className="font-medium">Stargate</div>
              <div className="text-xs text-gray-400">Portal to higher realms</div>
            </div>
          </div>
        </Link>
      </motion.div>
    </div>
  );
}

// Sub-component: Temples dropdown grouping EKAM and New Jerusalem
function TemplesDropdown({ currentPath }: { currentPath: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);
  const isActive = ["/temples", "/ekam", "/new-jerusalem"].some(p => currentPath === p || currentPath.startsWith(p + "/"));

  return (
    <div
      ref={ref}
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <motion.button
        className={`relative px-4 py-2 rounded-xl transition-all duration-300 group cursor-pointer flex items-center gap-2 ${
          isActive ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg' : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
        }`}
        whileHover={{ scale: 1.05, y: -2 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(v => !v)}
      >
        <span className="text-lg">🕍</span>
        <span className="font-medium hidden md:inline">Temples</span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="hidden md:inline"
        >
          ▼
        </motion.span>
      </motion.button>

      {/* Dropdown menu */}
      <motion.div
        initial={false}
        animate={{ opacity: open ? 1 : 0, y: open ? 0 : -8, pointerEvents: open ? 'auto' : 'none' }}
        transition={{ duration: 0.15 }}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="absolute right-0 mt-2 w-60 bg-black/80 backdrop-blur-md border border-purple-500/30 rounded-2xl overflow-hidden shadow-xl z-[300]"
      >
        <Link href="/temples" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/temples') ? 'text-white' : 'text-gray-300'}`}>
            <span>🏛️</span>
            <div>
              <div className="font-medium">Temples Home</div>
              <div className="text-xs text-gray-400">Overview of sacred spaces</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/ekam" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/ekam') ? 'text-white' : 'text-gray-300'}`}>
            <span>🕉️</span>
            <div>
              <div className="font-medium">EKAM</div>
              <div className="text-xs text-gray-400">Temple of One Consciousness</div>
            </div>
          </div>
        </Link>
        <Link href="/new-jerusalem" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/new-jerusalem') ? 'text-white' : 'text-gray-300'}`}>
            <span>🌈</span>
            <div>
              <div className="font-medium">New Jerusalem</div>
              <div className="text-xs text-gray-400">Sacred Geometry Museum</div>
            </div>
          </div>
        </Link>
      </motion.div>
    </div>
  );
}

// Sub-component: Dashboard dropdown grouping Wallet, Mining, Status
function DashboardDropdown({ currentPath }: { currentPath: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (!ref.current) return;
      if (!ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);
  const isActive = ["/", "/wallet", "/mining", "/miner", "/mining-status"].some(p =>
    p === "/" ? currentPath === "/" : currentPath === p || currentPath.startsWith(p + "/")
  );

  return (
    <div
      ref={ref}
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <motion.button
        className={`relative px-4 py-2 rounded-xl transition-all duration-300 group cursor-pointer flex items-center gap-2 ${
          isActive ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg' : 'bg-white/10 text-gray-300 hover:bg-white/20 hover:text-white'
        }`}
        whileHover={{ scale: 1.05, y: -2 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(v => !v)}
      >
        <span className="text-lg">📊</span>
        <span className="font-medium hidden md:inline">Dashboard</span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="hidden md:inline"
        >
          ▼
        </motion.span>

        {/* Tooltip */}
        <motion.div
          className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-black/90 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap pointer-events-none"
          initial={{ opacity: 0, scale: 0.8 }}
          whileHover={{ opacity: 1, scale: 1 }}
        >
          Real-time ZION Core Monitoring
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-black/90" />
        </motion.div>
      </motion.button>

      {/* Dropdown menu */}
      <motion.div
        initial={false}
        animate={{ opacity: open ? 1 : 0, y: open ? 0 : -8, pointerEvents: open ? 'auto' : 'none' }}
        transition={{ duration: 0.15 }}
        className="absolute right-0 mt-2 w-64 bg-black/80 backdrop-blur-md border border-purple-500/30 rounded-2xl overflow-hidden shadow-xl z-[300]"
      >
        <Link href="/" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath === '/' ? 'text-white' : 'text-gray-300'}`}>
            <span>📊</span>
            <div>
              <div className="font-medium">Main Dashboard</div>
              <div className="text-xs text-gray-400">Overview & system status</div>
            </div>
          </div>
        </Link>
        <div className="h-px bg-white/10" />
        <Link href="/wallet" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/wallet') ? 'text-white' : 'text-gray-300'}`}>
            <span>💰</span>
            <div>
              <div className="font-medium">Wallet</div>
              <div className="text-xs text-gray-400">ZION Wallet & Lightning</div>
            </div>
          </div>
        </Link>
        <Link href="/miner" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/miner') || currentPath.startsWith('/mining') ? 'text-white' : 'text-gray-300'}`}>
            <span>⛏️</span>
            <div>
              <div className="font-medium">Mining</div>
              <div className="text-xs text-gray-400">Mining Control Center</div>
            </div>
          </div>
        </Link>
        <Link href="/mining-status" className="block">
          <div className={`px-4 py-3 text-sm flex items-center gap-3 hover:bg-white/10 ${currentPath.startsWith('/mining-status') ? 'text-white' : 'text-gray-300'}`}>
            <span>📈</span>
            <div>
              <div className="font-medium">Status</div>
              <div className="text-xs text-gray-400">Mining & System Status</div>
            </div>
          </div>
        </Link>
      </motion.div>
    </div>
  );
}