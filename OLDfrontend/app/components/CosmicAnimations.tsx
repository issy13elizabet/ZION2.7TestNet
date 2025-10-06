"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface ParticleProps {
  color?: string;
  size?: number;
  speed?: number;
  count?: number;
}

export function CosmicParticles({ 
  color = "cyan", 
  size = 1, 
  speed = 15, 
  count = 50 
}: ParticleProps) {
  const [dimensions, setDimensions] = useState({ width: 1920, height: 1080 });

  useEffect(() => {
    setDimensions({
      width: window.innerWidth,
      height: window.innerHeight
    });
    
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getColorClass = (color: string) => {
    switch (color) {
      case 'gold': return 'bg-yellow-400';
      case 'purple': return 'bg-purple-400';
      case 'cyan': return 'bg-cyan-400';
      case 'pink': return 'bg-pink-400';
      case 'green': return 'bg-green-400';
      default: return 'bg-white';
    }
  };

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className={`absolute rounded-full opacity-70 ${getColorClass(color)}`}
          style={{
            width: `${size}px`,
            height: `${size}px`,
          }}
          animate={{
            x: [
              Math.random() * dimensions.width,
              Math.random() * dimensions.width,
              Math.random() * dimensions.width,
            ],
            y: [
              Math.random() * dimensions.height,
              Math.random() * dimensions.height,
              Math.random() * dimensions.height,
            ],
            rotate: [0, 360, 720],
            scale: [0.5, 1, 0.5],
          }}
          transition={{
            duration: speed + Math.random() * 10,
            repeat: Infinity,
            ease: "linear",
          }}
          initial={{
            x: Math.random() * dimensions.width,
            y: Math.random() * dimensions.height,
          }}
        />
      ))}
    </div>
  );
}

interface FloatingElementProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  amplitude?: number;
}

export function FloatingElement({ 
  children, 
  delay = 0, 
  duration = 4, 
  amplitude = 10 
}: FloatingElementProps) {
  return (
    <motion.div
      animate={{
        y: [0, -amplitude, 0],
        rotate: [-1, 1, -1],
      }}
      transition={{
        duration,
        repeat: Infinity,
        ease: "easeInOut",
        delay,
      }}
    >
      {children}
    </motion.div>
  );
}

interface PulsingGlowProps {
  children: React.ReactNode;
  color?: string;
  intensity?: number;
}

export function PulsingGlow({ 
  children, 
  color = "purple", 
  intensity = 1 
}: PulsingGlowProps) {
  const glowColor = {
    purple: "shadow-purple-500/50",
    cyan: "shadow-cyan-500/50",
    gold: "shadow-yellow-500/50",
    green: "shadow-green-500/50",
    pink: "shadow-pink-500/50",
  }[color] || "shadow-purple-500/50";

  return (
    <motion.div
      className={`${glowColor}`}
      animate={{
        boxShadow: [
          `0 0 ${10 * intensity}px ${color === 'purple' ? '#8b5cf6' : '#06b6d4'}`,
          `0 0 ${30 * intensity}px ${color === 'purple' ? '#8b5cf6' : '#06b6d4'}`,
          `0 0 ${10 * intensity}px ${color === 'purple' ? '#8b5cf6' : '#06b6d4'}`,
        ],
      }}
      transition={{
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      {children}
    </motion.div>
  );
}

interface TypewriterProps {
  text: string;
  speed?: number;
  className?: string;
}

export function TypewriterText({ 
  text, 
  speed = 50, 
  className = "" 
}: TypewriterProps) {
  const [displayText, setDisplayText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    }
  }, [currentIndex, text, speed]);

  return (
    <span className={className}>
      {displayText}
      <motion.span
        animate={{ opacity: [1, 0, 1] }}
        transition={{ duration: 1, repeat: Infinity }}
        className="inline-block w-0.5 h-5 bg-current ml-1"
      />
    </span>
  );
}

interface CosmicButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'cosmic';
  className?: string;
}

export function CosmicButton({ 
  children, 
  onClick, 
  variant = 'primary',
  className = "" 
}: CosmicButtonProps) {
  const variants = {
    primary: "bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700",
    secondary: "bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700",
    cosmic: "bg-gradient-to-r from-yellow-500 via-purple-500 to-pink-500 hover:from-yellow-600 hover:via-purple-600 hover:to-pink-600"
  };

  return (
    <motion.button
      className={`${variants[variant]} text-white font-bold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 ${className}`}
      onClick={onClick}
      whileHover={{ 
        scale: 1.05,
        boxShadow: "0 10px 25px rgba(0,0,0,0.3)"
      }}
      whileTap={{ scale: 0.95 }}
      animate={{
        backgroundPosition: ["0%", "100%", "0%"],
      }}
      transition={{
        backgroundPosition: {
          duration: 3,
          repeat: Infinity,
          ease: "linear"
        }
      }}
    >
      <FloatingElement amplitude={2}>
        {children}
      </FloatingElement>
    </motion.button>
  );
}