"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";

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
      href: "/",
      label: "Dashboard",
      icon: "📊",
      description: "Real-time ZION Core Monitoring"
    },
    {
      href: "/explorer",
      label: "Explorer",
      icon: "🌐",
      description: "Blockchain Explorer"
    },
    {
      href: "/miner",
      label: "Mining",
      icon: "⛏️",
      description: "Mining Control Center"
    },
    {
      href: "/mining-status",
      label: "Status",
      icon: "📈",
      description: "Mining & System Status"
    },
    {
      href: "/wallet",
      label: "Wallet",
      icon: "💰",
      description: "ZION Wallet & Lightning"
    },
    {
      href: "/ai",
      label: "AI Systems",
      icon: "🤖",
      description: "AI Module Control"
    }
  ];

  return (
    <motion.nav 
      className="fixed top-6 left-6 right-6 z-50 bg-black/20 backdrop-blur-md border border-purple-500/30 rounded-2xl p-4"
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
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-3">
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