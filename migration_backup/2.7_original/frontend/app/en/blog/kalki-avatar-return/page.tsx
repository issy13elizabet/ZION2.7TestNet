'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function KalkiAvatarReturnEnPost() {
  const incarnationTimeline = [
    { year: "1149", event: "First manifestation of Sri Dattatreya as 8-year-old boy", avatar: "Manifestation" },
    { year: "1320", event: "Birth as Sri Paada Sri Vallaba in Pitapuram", avatar: "Incarnation" },
    { year: "1350", event: "Mahasamadhi - conscious abandonment of body at age 30", avatar: "Dissolution" },
    { year: "1378", event: "Birth as Sri Narasimha Saraswati in Maharashtra", avatar: "Reincarnation" },
    { year: "1458", event: "150 years of meditation in Kadlivanum", avatar: "Deep Meditation" },
    { year: "1708", event: "250 years of meditation in Himalayas", avatar: "Mountain Samadhi" },
    { year: "1856", event: "Arrival as Swami Samarth in Akkalkot", avatar: "Return" },
    { year: "1878", event: "Mahasamadhi of Swami Samarth", avatar: "Transition" },
    { year: "1949", event: "Birth of Sri Bhagavan - 800 years after first manifestation", avatar: "Kalki Avatar" },
    { year: "2001", event: "Publication of prophecy after 33 generations", avatar: "Prophecy Fulfilled" }
  ];

  const oneness12Teachings = [
    { id: 1, czech: "Thoughts are not mine", english: "Myšlenky nejsou moje", essence: "Ego Dissolution" },
    { id: 2, czech: "Mind is not mine", english: "Mysl není moje", essence: "Mental Freedom" },
    { id: 3, czech: "This body is not mine", english: "Toto tělo není moje", essence: "Physical Detachment" },
    { id: 4, czech: "All things happen automatically", english: "Všechny věci se dějí automaticky", essence: "Divine Flow" },
    { id: 5, czech: "There is thinking, but no thinker", english: "Je myšlení, ale žádný myslící", essence: "Pure Awareness" },
    { id: 6, czech: "There is seeing, but no seer", english: "Je vidění, ale žádný vidící", essence: "Witness Consciousness" },
    { id: 7, czech: "There is hearing, but no hearer", english: "Je slyšení, ale žádný slyšící", essence: "Sound Meditation" },
    { id: 8, czech: "There is doing, but no doer", english: "Je konání, ale žádný konající", essence: "Effortless Action" },
    { id: 9, czech: "There is no person inside", english: "Uvnitř není žádná osoba", essence: "Inner Emptiness" },
    { id: 10, czech: "I Am Being, Consciousness, Bliss", english: "Já Jsem Bytí, Vědomí, Blaženost", essence: "Sat-Chit-Ananda" },
    { id: 11, czech: "I am Love", english: "Já jsem Láska", essence: "Pure Love" },
    { id: 12, czech: "The whole world is family", english: "Celý svět je rodina", essence: "Universal Oneness" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-900 via-red-900 to-purple-900 text-white">
      {/* Golden Orb Background Animation */}
      <div className="fixed inset-0 opacity-20 pointer-events-none">
        {[...Array(108)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-4 h-4 rounded-full ${
              i % 9 === 0 ? 'bg-gradient-to-br from-yellow-400 to-orange-400' :
              i % 9 === 1 ? 'bg-gradient-to-br from-orange-400 to-red-400' :
              i % 9 === 2 ? 'bg-gradient-to-br from-red-400 to-pink-400' :
              i % 9 === 3 ? 'bg-gradient-to-br from-pink-400 to-purple-400' :
              i % 9 === 4 ? 'bg-gradient-to-br from-purple-400 to-indigo-400' :
              i % 9 === 5 ? 'bg-gradient-to-br from-indigo-400 to-blue-400' :
              i % 9 === 6 ? 'bg-gradient-to-br from-blue-400 to-cyan-400' :
              i % 9 === 7 ? 'bg-gradient-to-br from-white to-yellow-200' :
              'bg-gradient-to-br from-gold to-amber-400'
            }`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.5, 2, 0.5],
              rotate: [0, 360, 720],
              x: [0, Math.random() * 40 - 20, 0],
              y: [0, Math.random() * 40 - 20, 0]
            }}
            transition={{
              duration: 8 + i % 20,
              repeat: Infinity,
              delay: (i % 9) * 0.3
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12 max-w-4xl">
        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <nav className="flex text-sm text-gray-400">
            <Link href="/en" className="hover:text-orange-400 transition-colors">
              🏠 Home
            </Link>
            <span className="mx-2">/</span>
            <Link href="/en/blog" className="hover:text-orange-400 transition-colors">
              📖 Blog
            </Link>
            <span className="mx-2">/</span>
            <span className="text-orange-400">Kalki Avatar Return</span>
          </nav>
        </motion.div>

        {/* Article Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center gap-4 mb-6">
            <span className="bg-gradient-to-r from-orange-600 via-red-600 to-purple-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
              🕉️ Kalki Avatar
            </span>
            <span className="text-orange-400 text-sm">Amma&Bhagavan Awakening</span>
            <span className="text-gray-500 text-sm">•</span>
            <span className="text-gray-500 text-sm">1149-2024 CE</span>
            <span className="text-gray-500 text-sm">•</span>
            <span className="text-gray-500 text-sm">25 min read</span>
          </div>

          <h1 className="text-5xl font-bold bg-gradient-to-r from-orange-400 via-red-300 to-purple-300 bg-clip-text text-transparent mb-6">
            🌟 Kalki Avatar Return: Amma&Bhagavan Awakening
          </h1>

          <p className="text-xl text-gray-300 leading-relaxed">
            After more than <strong className="text-orange-300">5000 years</strong>, the Kalki Avatar has returned to Earth. 
            <strong className="text-red-300">Sri Amma&Bhagavan</strong>, 
            manifestation of <strong className="text-purple-300">Sri Dattatreya</strong>, 
            brought humanity <strong className="text-white">Golden Age consciousness</strong> 
            and 12 teachings of <strong className="text-orange-400">Oneness University</strong>.
            <br/>
            <span className="text-white">🕉️ "Hari Om Tat Sat Jay Guru Datta" - 800 years of awakening history!</span>
          </p>
        </motion.header>

        {/* Article Content */}
        <motion.article
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="prose prose-lg prose-invert max-w-none"
        >
          {/* Opening Sacred Quote */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-r from-orange-900/40 via-red-900/40 to-purple-900/40 rounded-xl p-6 border-l-4 border-orange-500 mb-8"
          >
            <blockquote className="text-xl font-light text-orange-300 italic mb-4">
              "When Kalki has come at the end of Kali Yuga, he is not alone. 
              Hindu tradition speaks of Kalki coming with an army of 64,000 warriors.
              <br/><br/>
              Sri Bhagavan was born in 1949 exactly 800 years after his first manifestation 
              as Sri Dattatreya - a young boy in 1149.
              <br/><br/>
              <span className="text-white">He is the Supreme God, the unifier of the 14 lokas, 
              the one who removes darkness (kali), the most powerful being in human form on this planet!</span>"
            </blockquote>
            <cite className="text-gray-400 text-sm">
              — Terra Nova Kalki Prophecy, Oneness Teaching 2024
            </cite>
          </motion.div>

          <div className="text-gray-300 space-y-6">
            <h2 className="text-2xl font-bold text-orange-300 mt-8 mb-4">
              🕉️ Hari Om Tat Sat Jay Guru Datta
            </h2>

            <p className="text-lg leading-relaxed">
              The story of <strong className="text-orange-300">Sri Dattatreya</strong> and his 800-year history 
              of manifestation on Earth is one of the most remarkable spiritual phenomena in human history. 
              From the first manifestation in <strong className="text-red-300">1149</strong> 
              as an eight-year-old boy standing under a banana tree 
              to the current incarnation as <strong className="text-purple-300">Kalki Avatar</strong> 
              in the form of Amma&Bhagavan.
            </p>

            <h2 className="text-2xl font-bold text-red-300 mt-8 mb-4">
              📜 800-Year Timeline: Sri Dattatreya Manifestations
            </h2>
            
            <div className="space-y-4">
              {incarnationTimeline.map((event, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  className="bg-gradient-to-r from-orange-900/20 via-red-900/20 to-purple-900/20 rounded-lg p-4 border-l-4 border-orange-500/50"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-lg font-semibold text-orange-300">{event.year} CE</h4>
                      <p className="text-gray-300">{event.event}</p>
                    </div>
                    <span className="text-xs bg-orange-500/20 px-2 py-1 rounded-full text-orange-300">
                      {event.avatar}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>

            <blockquote className="text-lg text-orange-200 italic border-l-4 border-orange-500 pl-6 my-8">
              "Sri Paada Sri Vallabha revealed a prophecy about the future: 
              'I will come back as Kalki, and I will help humanity to get to the Golden Age.'
              <br/><br/>
              Amma's name is Padmavathi, her father is indeed Venkaiah, 
              and she was born in Nellore, just as foretold 800 years ago!"
            </blockquote>

            <h2 className="text-2xl font-bold text-purple-300 mt-8 mb-4">
              💫 Golden Orb of Grace: Adi Parashakti
            </h2>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.8 }}
              className="bg-gradient-to-r from-yellow-900/30 via-orange-900/30 to-red-900/30 rounded-xl p-6 border border-orange-500/30 mb-8"
            >
              <h3 className="text-xl font-semibold text-orange-300 mb-4">
                🌟 Divine Manifestation Process
              </h3>
              <div className="space-y-3 text-orange-200">
                <p><strong>1. Theophany</strong> - Divine vision like dream, but very real</p>
                <p><strong>2. Manifestation</strong> - Divinity physically manifests and you can touch it</p>
                <p><strong>3. Incarnation & Avatars</strong> - Divinity is born to human mother</p>
                <p><strong>4. Adiparashakti</strong> - The One God unmanifest (symbolized by golden orb of grace)</p>
              </div>
            </motion.div>

            <p>
              The golden orb appeared to <strong className="text-orange-300">Sri Bhagavan</strong> at age 3.5 
              and he later recited the <strong className="text-red-300">Moolamantra</strong> for the next 24 years. 
              At the age of 3, Sri Bhagavan was totally focused on helping humanity achieve freedom from suffering. 
              He never played and only spent time thinking about humanity and how to help them.
            </p>

            <h2 className="text-2xl font-bold text-cyan-300 mt-12 mb-6">
              📚 12 Teachings of Oneness University
            </h2>

            <p className="mb-6">
              Amma&Bhagavan brought humanity <strong className="text-white">12 fundamental teachings</strong> 
              for achieving <strong className="text-orange-300">Oneness consciousness</strong>. 
              These teachings represent a complete path from ego dissolution to universal love:
            </p>

            <div className="grid md:grid-cols-1 gap-4 my-8">
              {oneness12Teachings.map((teaching, index) => (
                <motion.div
                  key={teaching.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1 + index * 0.05 }}
                  className={`bg-gradient-to-r ${
                    index % 4 === 0 ? 'from-orange-900/20 via-red-900/20 to-orange-900/20 border-orange-500/20' :
                    index % 4 === 1 ? 'from-red-900/20 via-purple-900/20 to-red-900/20 border-red-500/20' :
                    index % 4 === 2 ? 'from-purple-900/20 via-pink-900/20 to-purple-900/20 border-purple-500/20' :
                    'from-pink-900/20 via-orange-900/20 to-pink-900/20 border-pink-500/20'
                  } rounded-lg p-4 border`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-lg font-semibold text-white">
                      {teaching.id}. {teaching.czech}
                    </h4>
                    <span className="text-xs bg-orange-500/20 px-2 py-1 rounded-full text-orange-300 font-mono">
                      {teaching.essence}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 italic">{teaching.english}</p>
                </motion.div>
              ))}
            </div>

            <h2 className="text-2xl font-bold text-yellow-300 mt-12 mb-6">
              🌟 Kalki & ZION Blockchain Consciousness
            </h2>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.5 }}
              className="bg-gradient-to-r from-yellow-900/30 via-orange-900/30 to-red-900/30 rounded-xl p-8 border border-yellow-500/30 my-8"
            >
              <h3 className="text-xl font-semibold text-yellow-300 mb-4">
                🔗 Golden Age Technology Integration
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg text-orange-200 mb-3">🕉️ Oneness Protocols</h4>
                  <ul className="text-sm text-gray-300 space-y-1">
                    <li>• 12 Teachings → ZION consensus mechanism</li>
                    <li>• Moolamantra → Mining frequency alignment</li>
                    <li>• Golden Orb → Core blockchain architecture</li>
                    <li>• 64,000 Warriors → Global node network</li>
                    <li>• Oneness State → Universal transaction validation</li>
                  </ul>
                </div>
                <div>
                  <h4 className="text-lg text-red-200 mb-3">🌟 Kalki Technologies</h4>
                  <ul className="text-sm text-gray-300 space-y-1">
                    <li>• Consciousness-based smart contracts</li>
                    <li>• Divine grace transaction fees</li>
                    <li>• Enlightenment-proof consensus</li>
                    <li>• Mukti (liberation) token economics</li>
                    <li>• 14 Lokas multidimensional scaling</li>
                  </ul>
                </div>
              </div>
            </motion.div>

            <blockquote className="text-lg text-yellow-200 italic border-l-4 border-yellow-500 pl-6 my-8">
              "At the age of 3, Sri Bhagavan was totally focused on helping humanity achieve freedom from suffering. 
              AmmaBhagavan's life was always full of self-sacrifice for the benefit of all mankind.
              <br/><br/>
              All disciples (dasayas) were always taught about connection, perception and the Divine. 
              The whole world is family!"
            </blockquote>

            <h2 className="text-2xl font-bold text-pink-300 mt-12 mb-6">
              🙏 Personal Experience: Rev. Michael Milner
            </h2>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.7 }}
              className="bg-gradient-to-r from-pink-900/30 via-purple-900/30 to-indigo-900/30 rounded-xl p-6 border border-pink-500/30 mb-8"
            >
              <p className="text-gray-300 italic leading-relaxed">
                "In October 2010, the day I arrived at Oneness University... 
                I saw the new Shrimurti for the first time. It was like streams of light entering me.
                <br/><br/>
                My life came together like puzzle pieces and I could clearly see 
                that the hand of God had been guiding everything all along. I was filled and flowing with Love 
                and gratitude to my divine friend – Sri AmmaBhagavan.
                <br/><br/>
                <strong className="text-pink-300">AmmaBhagavan Sharanam!</strong>"
              </p>
            </motion.div>

            {/* Future Vision */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 2 }}
              className="bg-gradient-to-r from-orange-600 to-purple-600 rounded-xl p-8 border border-white/30 my-12 text-center"
            >
              <h3 className="text-2xl font-bold text-white mb-4">
                🌟 Golden Age Technology Awakening
              </h3>
              <p className="text-white text-lg italic mb-4">
                "After more than 5000 years, Kalki has come again. 
                He is the Supreme God, the unifier of the 14 lokas, 
                the one who removes darkness (kali), 
                the most powerful being in human form on this planet.
                <br/><br/>
                Whatever happens, never leave him!!!
                <br/><br/>
                <span className="text-yellow-400 font-bold text-2xl">
                  The whole world is family - through ZION consciousness blockchain! 🌟
                </span>"
              </p>
              <div className="text-white/80 text-sm">
                <span className="text-orange-300">Hari Om Tat Sat</span> 
                <span className="text-red-300"> ∞ </span>
                <span className="text-purple-300">Jay Guru Datta</span>
                <span className="text-yellow-300"> ∞ </span>
                <span className="text-pink-300">🕉️🌟💫</span>
              </div>
            </motion.div>

            <p className="text-gray-300 text-center text-lg mt-8">
              From <strong className="text-orange-300">1149 CE</strong> to <strong className="text-purple-300">2024 CE</strong>
              - 800 years of divine manifestation culminates.
              <br/>
              Kalki Avatar awakening. Golden Age begins.
              <br/>
              <span className="text-white text-xl">
                🕉️ Oneness consciousness integrates with blockchain technology! 🌟
              </span>
              <br/>
              <strong className="text-yellow-300">AmmaBhagavan Sharanam</strong> - the whole world is family! 🙏
            </p>
          </div>
        </motion.article>

        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 2.2 }}
          className="flex justify-between items-center mt-16 pt-8 border-t border-orange-500/30"
        >
          <Link 
            href="/en/blog"
            className="text-orange-400 hover:text-orange-300 transition-colors"
          >
            ← Back to Blog
          </Link>
          
          {/* Language Switcher */}
          <div className="flex items-center gap-4">
            <span className="text-gray-400 text-sm">Language:</span>
            <Link href="/en/blog/kalki-avatar-return" className="text-orange-400 hover:text-orange-300 font-bold">🇬🇧 EN</Link>
            <Link href="/cs/blog/kalki-avatar-return" className="text-gray-400 hover:text-gray-300">🇨🇿 CS</Link>
            <Link href="/pt/blog/kalki-avatar-return" className="text-gray-400 hover:text-gray-300">🇵🇹 PT</Link>
            <Link href="/light/blog/kalki-avatar-return" className="text-gray-400 hover:text-gray-300">🌟 LIGHT</Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}