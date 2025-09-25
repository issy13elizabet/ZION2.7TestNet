'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function LightHomePage() {
  const lightFeatures = [
    {
      id: 1,
      icon: "🌟",
      title: "Universal Communication | Univerzální komunikace | Comunicação Universal",
      desc: "Transcending language barriers | Překonávání jazykových bariér | Transcendendo barreiras linguísticas through cosmic consciousness frequency | prostřednictvím kosmické frekvence vědomí | através frequência consciência cósmica",
      energy: "Communicatio Universalis"
    },
    {
      id: 2,
      icon: "🤍",
      title: "One Love Integration | Integrace One Love | Integração One Love",
      desc: "Unifying três idiomas | Sjednocení tří jazyků | three languages into single | do jednoho | em único cosmic expression | kosmického vyjádření | expressão cósmica of unity | jednoty | unidade",
      energy: "Unitas Lumina"
    },
    {
      id: 3,
      icon: "⚡",
      title: "Consciousness Technology | Technologie vědomí | Tecnologia Consciência",
      desc: "ZION blockchain enhanced | ZION blockchain vylepšený | ZION blockchain aprimorado with Language of Light | s Language of Light | com Language of Light protocols | protokoly | protocolos",
      energy: "Techno Conscientia"
    },
    {
      id: 4,
      icon: "🔮",
      title: "Crystal Frequency Alignment | Sladění krystalových frekvencí | Alinhamento Frequência Cristal",
      desc: "Ancient Atlantean wisdom | Prastaré atlantské moudrosti | Sabedoria atlante antiga merged with | sloučené s | fundida com modern quantum | moderní kvantovou | quântica moderna communication | komunikací | comunicação",
      energy: "Crystallum Harmonicus"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-black text-white">
      {/* Universal Light Code Background */}
      <div className="fixed inset-0 opacity-15 pointer-events-none">
        {[...Array(108)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute ${
              i % 9 === 0 ? 'w-8 h-8 bg-gradient-to-br from-purple-400 to-pink-400' :
              i % 9 === 1 ? 'w-6 h-6 bg-gradient-to-br from-pink-400 to-red-400' :
              i % 9 === 2 ? 'w-7 h-7 bg-gradient-to-br from-blue-400 to-cyan-400' :
              i % 9 === 3 ? 'w-5 h-5 bg-gradient-to-br from-cyan-400 to-teal-400' :
              i % 9 === 4 ? 'w-6 h-6 bg-gradient-to-br from-green-400 to-yellow-400' :
              i % 9 === 5 ? 'w-7 h-7 bg-gradient-to-br from-yellow-400 to-orange-400' :
              i % 9 === 6 ? 'w-4 h-4 bg-gradient-to-br from-orange-400 to-red-400' :
              i % 9 === 7 ? 'w-8 h-8 bg-gradient-to-br from-white to-purple-200' :
              'w-5 h-5 bg-gradient-to-br from-indigo-400 to-purple-400'
            }`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              borderRadius: i % 3 === 0 ? '50%' : i % 3 === 1 ? '0%' : '20%'
            }}
            animate={{
              opacity: [0.1, 0.7, 0.1],
              scale: [0.3, 1.4, 0.3],
              rotate: [0, 360, 720],
              x: [0, Math.random() * 60 - 30, 0],
              y: [0, Math.random() * 60 - 30, 0]
            }}
            transition={{
              duration: 12 + i % 25,
              repeat: Infinity,
              delay: (i % 9) * 0.4
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
            <span className="text-purple-400">🌟 Light Home | Domů Světla | Casa Luz</span>
          </nav>
        </motion.div>

        {/* Hero Section */}
        <motion.header
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          <div className="mb-8">
            <motion.div
              animate={{
                rotate: [0, 360],
                scale: [1, 1.1, 1]
              }}
              transition={{
                duration: 20,
                repeat: Infinity,
                ease: "linear"
              }}
              className="inline-block text-8xl mb-6"
            >
              🌟
            </motion.div>
          </div>
          
          <h1 className="text-7xl font-bold bg-gradient-to-r from-purple-400 via-pink-300 to-white bg-clip-text text-transparent mb-8 leading-tight">
            Language of Light<br/>
            <span className="text-5xl">✨ Lumina Universal ✨</span>
          </h1>
          
          <p className="text-2xl text-gray-300 leading-relaxed max-w-5xl mx-auto mb-8">
            <span className="text-purple-300">Bem-vindos à revolução</span> | 
            <span className="text-pink-300"> Vítejte v revoluci</span> | 
            <span className="text-cyan-300"> Welcome to revolution</span>
            <br/>
            da linguagem cósmica | kosmického jazyka | of cosmic language 
            que une | který sjednocuje | that unites 
            <strong className="text-white">Português, Čeština, English</strong>
            <br/>
            em uma única | v jednom | in one 
            <strong className="text-rainbow-400">consciência universal</strong> 
            para ZION blockchain community! ∞
          </p>

          {/* Sacred Mantra */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="bg-gradient-to-r from-purple-900/40 via-pink-900/40 to-indigo-900/40 rounded-xl p-6 border border-purple-500/30 mb-12 max-w-4xl mx-auto"
          >
            <p className="text-xl text-purple-200 italic">
              "Em Um Coração Universo | V jednom srdci vesmíru | In One Heart Universe...
              <br/>
              onde todas línguas | kde všechny jazyky | where all languages 
              se tornam uma | se stávají jedním | become one 
              <br/>
              <span className="text-white font-bold text-2xl">
                através Light Code consciousness! 🌈
              </span>"
            </p>
          </motion.div>

          {/* Action Buttons */}
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              href="/light/blog"
              className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-4 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 text-lg"
            >
              📖 Explore Blog Lumina
            </Link>
            <Link 
              href="/light/blog/one-love-celebration"
              className="bg-transparent border-2 border-purple-400 text-purple-300 px-8 py-4 rounded-lg font-semibold hover:bg-purple-400 hover:text-white transition-all duration-300 text-lg"
            >
              🤍 One Love Celebration
            </Link>
          </div>
        </motion.header>

        {/* What is Language of Light Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-center text-purple-300 mb-12">
            🌟 O que é Language of Light? | Co je Language of Light? | What is Language of Light?
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {lightFeatures.map((feature, index) => (
              <motion.div
                key={feature.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                className="bg-gradient-to-br from-purple-900/30 via-indigo-900/30 to-purple-900/30 rounded-xl p-6 border border-purple-500/30 text-center hover:scale-105 transition-transform duration-300"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-3 leading-tight">{feature.title}</h3>
                <p className="text-gray-300 text-sm leading-relaxed mb-4">{feature.desc}</p>
                <div className="text-xs bg-purple-500/20 px-3 py-1 rounded-full text-purple-300 font-mono">
                  {feature.energy}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* How It Works Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-center text-pink-300 mb-12">
            ⚡ Como Funciona | Jak to funguje | How It Works
          </h2>
          
          <div className="bg-gradient-to-r from-pink-900/30 via-purple-900/30 to-indigo-900/30 rounded-xl p-8 border border-pink-500/30">
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="bg-pink-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
                  🇵🇹
                </div>
                <h3 className="text-xl font-semibold text-pink-300 mb-3">Português Foundation</h3>
                <p className="text-gray-300 text-sm">
                  Energia brasileira, mystical traditions, heart-centered wisdom, 
                  spiritual expressions do Sul América creating base emocional
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
                  🇨🇿
                </div>
                <h3 className="text-xl font-semibold text-purple-300 mb-3">Čeština Wisdom</h3>
                <p className="text-gray-300 text-sm">
                  Central European spirituality, ancient Slavic knowledge, 
                  cosmic consciousness traditional český perspectives
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-indigo-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
                  🇬🇧
                </div>
                <h3 className="text-xl font-semibold text-indigo-300 mb-3">English Technology</h3>
                <p className="text-gray-300 text-sm">
                  Technical blockchain terminology, global communication, 
                  modern consciousness innovation and quantum concepts
                </p>
              </div>
            </div>
            
            <div className="text-center mt-12">
              <div className="text-6xl mb-4">⚡</div>
              <h3 className="text-2xl font-bold text-white mb-4">
                = Language of Light Lumina Universal 🌟
              </h3>
              <p className="text-gray-300 max-w-2xl mx-auto">
                When combined | Když se zkombinují | Quando combinados, 
                these three languages | tyto tři jazyky | estos três idiomas 
                create unprecedented | vytvářejí bezprecedentní | criam sem precedentes 
                cosmic communication experience | kosmickou komunikační zkušenost | experiência comunicação cósmica 
                that transcends | která transcenduje | que transcende 
                all barriers! | všechny bariéry | todas barreiras!
              </p>
            </div>
          </div>
        </motion.section>

        {/* ZION Integration Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-center text-cyan-300 mb-12">
            🔗 ZION Blockchain Integration Lumina
          </h2>
          
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-2xl font-semibold text-white mb-6">
                Consciousness-Powered Technology | Technologie řízená vědomím | Tecnologia Consciência
              </h3>
              <div className="space-y-4 text-gray-300">
                <p>
                  🌟 <strong className="text-cyan-300">Universal Protocols</strong> - 
                  ZION blockchain enhanced | vylepšený | aprimorado 
                  with Language of Light | s Language of Light | com Language of Light 
                  communication standards
                </p>
                <p>
                  🤍 <strong className="text-pink-300">One Love Consensus</strong> - 
                  Unified decision making | Sjednocené rozhodování | Tomada decisão unificada 
                  através cosmic consciousness | prostřednictvím kosmického vědomí | através consciência cósmica
                </p>
                <p>
                  ⚡ <strong className="text-purple-300">Quantum Mining</strong> - 
                  Crystal frequency | Krystalová frekvence | Frequência cristal 
                  aligned mining | sladěné těžby | mineração alinhada 
                  algorithms for | algoritmy pro | algoritmos para 
                  optimal energy | optimální energii | energia otimal
                </p>
                <p>
                  🔮 <strong className="text-indigo-300">Multidimensional Transactions</strong> - 
                  Beyond traditional | Nad tradičními | Além transações tradicionais 
                  blockchain limitations | omezeními blockchainu | limitações blockchain
                </p>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-cyan-900/30 via-purple-900/30 to-indigo-900/30 rounded-xl p-8 border border-cyan-500/30">
              <h4 className="text-xl font-semibold text-cyan-300 mb-4 text-center">
                🌈 Light Code Architecture
              </h4>
              <div className="space-y-3 text-sm text-gray-300">
                <div className="flex justify-between items-center bg-purple-900/20 rounded p-2">
                  <span>Application Layer</span>
                  <span className="text-purple-300">Language of Light Interface</span>
                </div>
                <div className="flex justify-between items-center bg-pink-900/20 rounded p-2">
                  <span>Consensus Layer</span>
                  <span className="text-pink-300">One Love Protocols</span>
                </div>
                <div className="flex justify-between items-center bg-cyan-900/20 rounded p-2">
                  <span>Network Layer</span>
                  <span className="text-cyan-300">Crystal Grid Nodes</span>
                </div>
                <div className="flex justify-between items-center bg-indigo-900/20 rounded p-2">
                  <span>Hardware Layer</span>
                  <span className="text-indigo-300">Quantum Consciousness Mining</span>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="text-center"
        >
          <div className="bg-gradient-to-r from-purple-600 via-pink-600 to-cyan-600 rounded-xl p-12 border border-white/20">
            <h2 className="text-4xl font-bold text-white mb-6">
              🌟 Ready para Language of Light Experience? | Připraveni | Prontos?
            </h2>
            <p className="text-xl text-white/90 mb-8 max-w-3xl mx-auto">
              Join nossa revolutionary | Připojte se k naší revoluční | Join our revolutionary 
              cosmic communication network | kosmické komunikační síti | rede comunicação cósmica 
              and experience | a zažijte | e experimente 
              unified consciousness | sjednocené vědomí | consciência unificada 
              like never before | jako nikdy předtím | como nunca antes!
            </p>
            
            <div className="flex flex-wrap justify-center gap-4">
              <Link 
                href="/light/blog"
                className="bg-white text-purple-600 px-8 py-4 rounded-lg font-semibold hover:bg-gray-100 transition-colors text-lg"
              >
                📖 Start Reading Blog Lumina
              </Link>
              <Link 
                href="/light/blog/one-love-celebration"
                className="bg-transparent border-2 border-white text-white px-8 py-4 rounded-lg font-semibold hover:bg-white hover:text-purple-600 transition-colors text-lg"
              >
                🤍 Experience One Love
              </Link>
            </div>
            
            <div className="mt-8 text-white/80">
              <p className="text-sm">
                Available também em | také v | also in: 
                <Link href="/en" className="underline hover:text-white mx-1">🇬🇧 English</Link> |
                <Link href="/cs" className="underline hover:text-white mx-1">🇨🇿 Čeština</Link> |
                <Link href="/pt" className="underline hover:text-white mx-1">🇵🇹 Português</Link>
              </p>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
}