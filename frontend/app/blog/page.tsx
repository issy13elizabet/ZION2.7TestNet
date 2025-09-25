'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function BlogPage() {
  const blogPosts = [
    {
      id: 'crystal-grid-activation',
      title: '💎 Crystal Grid 144: ZION Network Activation',
      excerpt: 'The ancient Atlantean crystals now power our blockchain consciousness through proof-of-work awakening...',
      date: '2025-09-22',
      category: 'Cosmic Technology',
      readTime: '8 min read',
      featured: true
    },
    {
      id: 'bolon-yokte-return',
      title: '🌟 Návrat Bolona Yokte (144,000 Avatárů Sjednocení)',
      excerpt: 'Vrátili jsme se jako Trojejediná podstata kosmu. Naše jméno je známé jako BOLON YOKTE - návrat 144,000 duší k založení Nového Jeruzaléma...',
      date: '2025-09-23',
      category: '144,000 Souls',
      readTime: '12 min read'
    },
    {
      id: 'genesis-awakening',
      title: '🌟 Genesis: The Era of Light',
      excerpt: 'When the earthly universe breaks through the waters in which it was submerged, a new universe will be born...',
      date: '2025-09-23',
      category: 'Oracle Prophecies',
      readTime: '5 min read'
    },
    {
      id: 'crystal-grid-activation',
      title: '💎 Crystal Grid 144: ZION Network Activation',
      excerpt: 'The ancient Atlantean crystals now power our blockchain consciousness through proof-of-work awakening...',
      date: '2025-09-22',
      category: 'Cosmic Technology',
      readTime: '8 min read',
      featured: true
    },
    {
      id: 'little-tibet-awakening',
      title: '🏔️ Little Tibet: Digital Dharma & Blockchain Nirvana',
      excerpt: 'From the sacred heights of the Himalayas to the digital peaks of ZION blockchain consciousness...',
      date: '2025-09-19',
      category: 'Sacred Mountains',
      readTime: '15 min read'
    },
    {
      id: 'one-love-celebration',
      title: '🤍 One Love Celebration: 22 Fields of Consciousness',
      excerpt: 'From 21.12.2012 to 21.12.2024 - witness the 12-year evolution of human consciousness through 22 dimensional fields...',
      date: '2025-09-20',
      category: 'One Love',
      readTime: '12 min read'
    },
    {
      id: 'atlantean-free-energy',
      title: '⚡ Atlantean Free Energy: ZION Power Revolution',
      excerpt: 'How ZION mining operations accidentally rediscovered the lost Atlantean free energy principles...',
      date: '2025-09-21',
      category: 'Free Energy',
      readTime: '10 min read'
    },
    {
      id: 'lightning-prophecy',
      title: '⚡ Lightning Network: Ancient Wisdom Meets Modern Technology',
      excerpt: 'The cosmic channels of instant payments fulfill ancient prophecies of instant communication across dimensions...',
      date: '2025-09-21',
      category: 'Technology',
      readTime: '4 min read'
    },
    {
      id: 'atomic-ceremonies',
      title: '🔄 Atomic Swaps: Cryptographic Marriage Rituals',
      excerpt: 'HTLC protocols echo ancient binding ceremonies, uniting separate blockchains in sacred digital matrimony...',
      date: '2025-09-20',
      category: 'Cosmic Tech',
      readTime: '6 min read'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white">
      {/* Cosmic Background */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.8, 1.5, 0.8]
            }}
            transition={{
              duration: 3 + Math.random() * 4,
              repeat: Infinity,
              delay: Math.random() * 3
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-400 via-purple-300 to-blue-300 bg-clip-text text-transparent mb-4">
            🔮 ZION Genesis Blog
          </h1>
          <motion.div 
            className="text-lg text-purple-400 mb-6 font-mono"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            "As the Oracle of Man becomes human, humans will become the Oracles of Light"
          </motion.div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Ancient wisdom meets modern blockchain technology. Explore the cosmic evolution of 
            human consciousness through decentralized digital enlightenment.
          </p>
        </motion.div>

        {/* Navigation Breadcrumb */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <nav className="flex text-sm text-gray-400">
            <Link href="/" className="hover:text-purple-300 transition-colors">
              🌌 Home
            </Link>
            <span className="mx-2">/</span>
            <span className="text-purple-300">📖 Genesis Blog</span>
          </nav>
        </motion.div>

        {/* Featured Post */}
        {blogPosts.filter(post => post.featured).map((post, index) => (
          <motion.article
            key={post.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-black/40 backdrop-blur-lg rounded-2xl p-8 border border-purple-500/30 mb-12"
          >
            <div className="flex items-center gap-4 mb-4">
              <span className="bg-gradient-to-r from-purple-600 to-violet-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                ✨ Featured
              </span>
              <span className="text-purple-400 text-sm">{post.category}</span>
              <span className="text-gray-500 text-sm">•</span>
              <span className="text-gray-500 text-sm">{post.date}</span>
              <span className="text-gray-500 text-sm">•</span>
              <span className="text-gray-500 text-sm">{post.readTime}</span>
            </div>
            
            <h2 className="text-3xl font-bold text-white mb-4 hover:text-purple-300 transition-colors">
              <Link href={`/blog/${post.id}`}>
                {post.title}
              </Link>
            </h2>
            
            <p className="text-gray-300 text-lg leading-relaxed mb-6">
              {post.excerpt}
            </p>
            
            <Link 
              href={`/blog/${post.id}`}
              className="inline-flex items-center bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700 text-white px-6 py-3 rounded-lg transition-all duration-300 transform hover:scale-105"
            >
              Read Full Prophecy ✨
            </Link>
          </motion.article>
        ))}

        {/* Blog Posts Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.filter(post => !post.featured).map((post, index) => (
            <motion.article
              key={post.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * (index + 3) }}
              className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 hover:transform hover:scale-105"
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="text-purple-400 text-xs bg-purple-900/30 px-2 py-1 rounded">
                  {post.category}
                </span>
                <span className="text-gray-500 text-xs">•</span>
                <span className="text-gray-500 text-xs">{post.readTime}</span>
              </div>
              
              <h3 className="text-xl font-semibold text-white mb-3 hover:text-purple-300 transition-colors">
                <Link href={`/blog/${post.id}`}>
                  {post.title}
                </Link>
              </h3>
              
              <p className="text-gray-400 text-sm leading-relaxed mb-4">
                {post.excerpt}
              </p>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-xs">{post.date}</span>
                <Link 
                  href={`/blog/${post.id}`}
                  className="text-purple-400 hover:text-purple-300 text-sm font-medium transition-colors"
                >
                  Read More →
                </Link>
              </div>
            </motion.article>
          ))}
        </div>

        {/* Oracle Wisdom Quote */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-16 text-center"
        >
          <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-2xl p-8 border border-purple-500/30">
            <blockquote className="text-2xl font-light text-purple-300 italic mb-4">
              "They will discover their gold in the language of light, through which many separate existences live in the elegance of Unity."
            </blockquote>
            <cite className="text-gray-400 text-sm">
              — Dohrman's Prophecy, Chapter 7: The Era of Light
            </cite>
          </div>
        </motion.div>

        {/* Cosmic Mantra */}
        <motion.div
          className="text-center mt-8"
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 5, repeat: Infinity }}
        >
          <p className="text-purple-400 text-sm">
            ⚡ Jai Ram Ram Ram Dohrman Oracle Genesis Ram Ram Ram Hanuman! ⚡
          </p>
        </motion.div>
      </div>
    </div>
  );
}