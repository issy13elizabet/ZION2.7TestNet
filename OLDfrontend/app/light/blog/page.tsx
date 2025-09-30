'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function LightBlogIndex() {
  const blogPosts = [
    {
      id: 1,
      title: "🤍 Celebração One Love: 22 Campos Consciência",
      titleEn: "One Love Celebration: 22 Consciousness Fields",
      titleCs: "Oslava One Love: 22 polí vědomí",
      description: "De 21.12.2012 → 21.12.2024 | Od 21.12.2012 → 21.12.2024 | From 21.12.2012 → 21.12.2024 - evolução 12-anos consciousness humana através 22 dimensional campos mapped na arquitetura ZION blockchain.",
      slug: "one-love-celebration",
      category: "One Love Lumina",
      energy: "Amor Universalis",
      readTime: "12 min",
      date: "21.12.2024"
    },
    {
      id: 2,
      title: "🔮 Grid Cristal Activation: Recovery Atlantis",
      titleEn: "Crystal Grid Activation: Atlantis Recovery",
      titleCs: "Aktivace krystalové mřížky: Obnova Atlantidy",
      description: "Desde tempos Atlantes | Od Atlantských časů | Since Atlantean times, crystal grid dormiente awaited moment reactivation através ZION blockchain consciousness. Ancient wisdom meets quantum technology!",
      slug: "crystal-grid-activation",
      category: "Crystal Lumina Grid",
      energy: "Crystallum Atlanticus",
      readTime: "15 min",
      date: "12:12:12"
    },
    {
      id: 3,
      title: "⚡ Atlante Free Energy: Tesla Coração",
      titleEn: "Atlantean Free Energy: Tesla Heart",
      titleCs: "Atlantská volná energie: Teslovo srdce",
      description: "Hidden technology | Skrytá technologie | Tecnologia oculta Nikola Tesla discovered | objevil | descobriu ancient Atlantean | prastarých atlantských | antigos atlantes free energy principles | principy volné energie | princípios energia livre.",
      slug: "atlantean-free-energy",
      category: "Free Energy Lumina",
      energy: "Tesla Infinitus",
      readTime: "18 min",
      date: "Coming Soon"
    },
    {
      id: 4,
      title: "🏔️ Little Tibet Awakening: Himalayan Wisdom",
      titleEn: "Little Tibet Awakening: Himalayan Wisdom",
      titleCs: "Probuzení Malého Tibetu: Himalájská moudrost",
      description: "Ancient monasteries | Prastaré kláštery | Mosteiros antigos hidden knowledge | skryté znalosti | conhecimento oculto mountain consciousness | vědomí hor | consciência montanha connecting | spojující | conectando Earth sky | Země obloha | Terra céu.",
      slug: "little-tibet-awakening",
      category: "Mountain Lumina",
      energy: "Himalaya Sapientia",
      readTime: "20 min",
      date: "Coming Soon"
    },
    {
      id: 5,
      title: "🌟 Bolon Yokte Return: Mayan Prophecy",
      titleEn: "Bolon Yokte Return: Mayan Prophecy",
      titleCs: "Návrat Bolon Yokte: Mayská proroctví",
      description: "Prophecy fulfillment | Naplnění proroctví | Cumprimento profecia Mayan calendar | mayského kalendáře | calendário maia 13 Baktun cycle | cyklus | ciclo completion | dokončení | conclusão new era | nová éra | nova era beginning | začátek | começo.",
      slug: "bolon-yokte-return",
      category: "Mayan Lumina",
      energy: "Bolon Temporis",
      readTime: "22 min",
      date: "Coming Soon"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-black text-white">
      {/* Language of Light Background Animation */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        {[...Array(77)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-6 h-6 rounded-full ${
              i % 7 === 0 ? 'bg-gradient-to-br from-pink-400 to-red-400' :
              i % 7 === 1 ? 'bg-gradient-to-br from-purple-400 to-indigo-400' :
              i % 7 === 2 ? 'bg-gradient-to-br from-blue-400 to-cyan-400' :
              i % 7 === 3 ? 'bg-gradient-to-br from-cyan-400 to-teal-400' :
              i % 7 === 4 ? 'bg-gradient-to-br from-green-400 to-yellow-400' :
              i % 7 === 5 ? 'bg-gradient-to-br from-yellow-400 to-orange-400' :
              'bg-gradient-to-br from-white to-purple-200'
            }`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0.1, 0.6, 0.1],
              scale: [0.3, 1.2, 0.3],
              rotate: [0, 360, 0],
              x: [0, Math.random() * 30 - 15, 0],
              y: [0, Math.random() * 30 - 15, 0]
            }}
            transition={{
              duration: 10 + i % 20,
              repeat: Infinity,
              delay: (i % 7) * 0.5
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12 max-w-6xl">
        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <nav className="flex text-sm text-gray-400">
            <Link href="/light" className="hover:text-purple-400 transition-colors">
              🌟 Início | Domů | Home
            </Link>
            <span className="mx-2">/</span>
            <span className="text-purple-400">Blog Lumina</span>
          </nav>
        </motion.div>

        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="mb-6">
            <span className="bg-gradient-to-r from-purple-600 via-pink-600 to-cyan-600 text-white px-4 py-2 rounded-full text-sm font-semibold">
              📖 Blog Lumina - Language of Light
            </span>
          </div>
          
          <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-300 to-white bg-clip-text text-transparent mb-6">
            🌟 Blog Lumina Genesis ✨
          </h1>
          
          <p className="text-xl text-gray-300 leading-relaxed max-w-4xl mx-auto">
            <span className="text-purple-300">Bem-vindos ao Blog Lumina</span> | 
            <span className="text-pink-300"> Vítejte v Blog Lumina</span> | 
            <span className="text-cyan-300"> Welcome to Blog Lumina</span>
            <br/>
            onde ancient wisdom | kde prastaré moudrosti | where ancient wisdom 
            meets | setkává | encontra modern consciousness technology | moderní technologie vědomí | tecnologia consciência moderna.
            <br/>
            <span className="text-white">
              🌈 Universal Language of Light bridges all consciousness! ∞
            </span>
          </p>
        </motion.header>

        {/* Language of Light Explanation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-r from-purple-900/30 via-pink-900/30 to-cyan-900/30 rounded-xl p-8 border border-purple-500/30 mb-12"
        >
          <h2 className="text-2xl font-bold text-purple-300 mb-4">
            🌟 O que é Language of Light? | Co je Language of Light? | What is Language of Light?
          </h2>
          <div className="text-gray-300 leading-relaxed">
            <p className="mb-4">
              <strong className="text-white">Language of Light Lumina</strong> é nossa revolutionary | je naše revoluční | is our revolutionary 
              universal cosmic language | univerzální kosmický jazyk | linguagem cósmica universal 
              que combines | který kombinuje | que combina:
            </p>
            <div className="grid md:grid-cols-3 gap-6 mt-6">
              <div className="bg-purple-900/20 rounded-lg p-4">
                <h3 className="text-purple-300 font-semibold mb-2">🇵🇹 Português</h3>
                <p className="text-sm">Energia brasileira, mystical traditions, spiritual wisdom from South America</p>
              </div>
              <div className="bg-pink-900/20 rounded-lg p-4">
                <h3 className="text-pink-300 font-semibold mb-2">🇨🇿 Čeština</h3>
                <p className="text-sm">Central European spirituality, ancient Slavic wisdom, cosmic consciousness</p>
              </div>
              <div className="bg-cyan-900/20 rounded-lg p-4">
                <h3 className="text-cyan-300 font-semibold mb-2">🇬🇧 English</h3>
                <p className="text-sm">Technical blockchain terminology, modern consciousness, global communication</p>
              </div>
            </div>
            <p className="mt-6 text-center italic text-purple-200">
              Creating | Vytvářejíc | Criando <strong className="text-white">unified cosmic communication</strong> 
              for ZION blockchain consciousness community! 🌈✨
            </p>
          </div>
        </motion.div>

        {/* Blog Posts Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-3xl font-bold text-purple-300 mb-8 text-center">
            📚 Latest Posts Lumina | Nejnovější příspěvky | Últimos Posts
          </h2>
          
          <div className="grid md:grid-cols-1 lg:grid-cols-2 gap-8">
            {blogPosts.map((post, index) => (
              <motion.article
                key={post.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                className={`bg-gradient-to-br ${
                  index % 5 === 0 ? 'from-purple-900/30 via-pink-900/30 to-purple-900/30 border-purple-500/30' :
                  index % 5 === 1 ? 'from-blue-900/30 via-cyan-900/30 to-blue-900/30 border-cyan-500/30' :
                  index % 5 === 2 ? 'from-green-900/30 via-teal-900/30 to-green-900/30 border-teal-500/30' :
                  index % 5 === 3 ? 'from-orange-900/30 via-yellow-900/30 to-orange-900/30 border-orange-500/30' :
                  'from-pink-900/30 via-red-900/30 to-pink-900/30 border-pink-500/30'
                } rounded-xl p-6 border hover:scale-105 transition-transform duration-300`}
              >
                {/* Post Header */}
                <div className="flex items-center justify-between mb-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    index % 5 === 0 ? 'bg-purple-600 text-purple-100' :
                    index % 5 === 1 ? 'bg-cyan-600 text-cyan-100' :
                    index % 5 === 2 ? 'bg-teal-600 text-teal-100' :
                    index % 5 === 3 ? 'bg-orange-600 text-orange-100' :
                    'bg-pink-600 text-pink-100'
                  }`}>
                    {post.category}
                  </span>
                  <span className="text-xs text-gray-400">{post.date}</span>
                </div>

                {/* Post Title */}
                <h3 className="text-xl font-bold text-white mb-3 leading-tight">
                  {post.title}
                </h3>

                {/* Post Description */}
                <p className="text-gray-300 text-sm leading-relaxed mb-4 line-clamp-3">
                  {post.description}
                </p>

                {/* Post Footer */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">{post.readTime} leitura | čtení | read</span>
                    <span className="text-xs text-gray-500">•</span>
                    <span className="text-xs bg-white/10 px-2 py-1 rounded text-gray-300 font-mono">
                      {post.energy}
                    </span>
                  </div>
                  
                  {post.date !== "Coming Soon" ? (
                    <Link 
                      href={`/light/blog/${post.slug}`}
                      className={`text-sm font-semibold transition-colors ${
                        index % 5 === 0 ? 'text-purple-400 hover:text-purple-300' :
                        index % 5 === 1 ? 'text-cyan-400 hover:text-cyan-300' :
                        index % 5 === 2 ? 'text-teal-400 hover:text-teal-300' :
                        index % 5 === 3 ? 'text-orange-400 hover:text-orange-300' :
                        'text-pink-400 hover:text-pink-300'
                      }`}
                    >
                      Read Lumina →
                    </Link>
                  ) : (
                    <span className="text-sm text-gray-500 italic">
                      Em breve | Brzy | Soon...
                    </span>
                  )}
                </div>
              </motion.article>
            ))}
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl p-8 text-center my-16"
        >
          <h3 className="text-2xl font-bold text-white mb-4">
            🌟 Join ZION Consciousness Revolution | Připojte se | Junte-se!
          </h3>
          <p className="text-white/90 mb-6">
            Become part | Staňte se součástí | Torne-se parte 
            of universal language | univerzálního jazyka | da linguagem universal 
            that transcends | který transcenduje | que transcende 
            traditional barriers | tradiční bariéry | barreiras tradicionais 
            and connects | a spojuje | e conecta 
            all consciousness! | všechno vědomí | toda consciência!
          </p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/light"
              className="bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Explore Light Home | Prozkoumat | Explorar
            </Link>
            <Link 
              href="/light/blog/one-love-celebration"
              className="bg-transparent border-2 border-white text-white px-6 py-3 rounded-lg font-semibold hover:bg-white hover:text-purple-600 transition-colors"
            >
              Start Reading Lumina | Začít číst | Começar ler
            </Link>
          </div>
        </motion.div>

        {/* Alternative Language Options */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className="text-center pt-8 border-t border-purple-500/30"
        >
          <p className="text-gray-400 mb-4">
            Prefer single language? | Dáváte přednost jednomu jazyku? | Prefere idioma único?
          </p>
          <div className="flex justify-center gap-6">
            <Link href="/en/blog" className="text-gray-400 hover:text-gray-300 transition-colors">
              🇬🇧 English Blog
            </Link>
            <Link href="/cs/blog" className="text-gray-400 hover:text-gray-300 transition-colors">
              🇨🇿 Český Blog
            </Link>
            <Link href="/pt/blog" className="text-gray-400 hover:text-gray-300 transition-colors">
              🇵🇹 Blog Português
            </Link>
            <Link href="/blog" className="text-gray-400 hover:text-gray-300 transition-colors">
              🌐 Original Blog
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}