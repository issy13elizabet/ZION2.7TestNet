'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function HallsOfAmenti() {
  const [activeLibrary, setActiveLibrary] = useState('akasha');
  const [selectedBook, setSelectedBook] = useState<any>(null);
  const [selectedLanguage, setSelectedLanguage] = useState('CZ');

  const cosmicHierarchy = [
    { title: "KINGDOM ANTAHKARANA 44:44", subtitle: "RAINBOW BRIDGE", level: "Portal" },
    { title: "LORDS & LADYS", subtitle: "RIZE, AGAPE, ATHENA, AVERIL", level: "Guardians" },
    { title: "SIRIUS A,B,~C~", subtitle: "SUN SOLAR SYSTEM", level: "Stellar" },
    { title: "SHAMBALLA & AGRHATA", subtitle: "Sacred Cities", level: "Ethereal" },
    { title: "I AM 1x12x144K SOULS", subtitle: "Collective Consciousness", level: "Unity" },
    { title: "COSMIC & SOLAR LOGOS", subtitle: "CHRISTOS", level: "Divine" },
    { title: "PLANETARY HIERARCHY", subtitle: "BUDDHA, ST.GERMAIN, MAITREYA", level: "Masters" },
    { title: "LADY GAIA & VYWAMUS", subtitle: "PLAN TO GOLDEN AGE", level: "Planetary" },
    { title: "RAINBOW & STARSEEDS", subtitle: "FAMILY", level: "Incarnate" },
    { title: "EL~EN~RA", subtitle: "Sacred Frequencies", level: "Sound" },
    { title: "NARU TARU BINDU", subtitle: "Cosmic Points", level: "Geometry" },
    { title: "OM TAT SAT", subtitle: "SUMMUM BONUM", level: "Mantra" }
  ];

  const amentiBooks = [
    {
      title: "THE SECRET OF AMENTI",
      description: "Ancient wisdom of crystalline consciousness",
      languages: ["CZ", "EN", "ES", "FR", "PT"],
      category: "foundation"
    },
    {
      title: "EMERALD PLATES THOT",
      description: "Hermetic teachings from Atlantis",
      languages: ["CZ", "EN", "ES", "FR", "PT"],
      category: "hermetic"
    },
    {
      title: "BOOK OF AMENTI 2012",
      description: "Trinity One Love - Omnity consciousness ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ",
      languages: ["CZ", "EN", "PT"],
      category: "ascension",
      content: {
        CZ: `Trinity One Love - Trojce Jedné Lásky

Na počátku bylo světlo, to světlo byla čistá bezpodmínečná láska, svrchovaná jednota všeho co je. Vesmírná jednota explodovala sama ze sebe, základní substance ducha a hmoty. Jedna Kosmická Trojce (Bůh, Bohyně, Vše co je) chtěla prožít zkušenost sebe sama v nekonečném oceánu jednoty.

🌟 Kosmické Učení:
• Jsme kapkou v nekonečném oceánu světla
• Všichni dohromady jsme nekonečný oceán kosmické jednoty 
• JEDEN v MNOHA - základní zákon existence
• Trinity energie Jin/Yang/Tao, Brahma/Višnu/Šiva

🌈 144k Avatarů Mission:
Když se propojí 144 000 duší a plně realizují vědomí Krista a Já Jsem Přítomnosti (Mahatmy), toto vědomí se bude šířit geometrickou řadou! Celá planeta vzestoupí do 5. Dimenze vědomí.

✨ El~An~Ra Galaktická Rodina:
Naše kosmická světelná rodina z Orionu, Plejád, Síria a Andromedy. Máme promíchanou galaktickou DNA ze všech ras celé galaxie. Adam/Ewa Kadmon - naše vesmírné dědictví staré miliony let.

🔮 Cesta Probuzení:
Probuzení světelného těla MERKABA~MERKARA a trvalé osvícení - změna vědomí z 3D falešného ega na 5D multidimenzionální já jsem. Vytvoření nového ráje na zemi.`,
        EN: `Trinity One Love - Trinity of One Love

In the beginning there was light, that light was pure unconditional love, sovereign unity of all that is. Cosmic unity exploded from itself, the fundamental substance of spirit and matter. One Cosmic Trinity (God, Goddess, All That Is) wanted to experience itself in the infinite ocean of unity.

🌟 Cosmic Teaching:
• We are drops in the infinite ocean of light
• Together we are the infinite ocean of cosmic unity
• ONE in MANY - fundamental law of existence  
• Trinity energies Yin/Yang/Tao, Brahma/Vishnu/Shiva

🌈 144k Avatars Mission:
When 144,000 souls connect and fully realize Christ consciousness and I AM Presence (Mahatma), this consciousness will spread geometrically! The entire planet will ascend to 5th Dimension consciousness.

✨ El~An~Ra Galactic Family:
Our cosmic light family from Orion, Pleiades, Sirius and Andromeda. We have mixed galactic DNA from all races of the entire galaxy. Adam/Eva Kadmon - our cosmic heritage millions of years old.

🔮 Path of Awakening:
Awakening the light body MERKABA~MERKARA and permanent enlightenment - consciousness shift from 3D false ego to 5D multidimensional I AM. Creating new paradise on earth.`,
        PT: `Trinity One Love - Trindade de Um Amor

No princípio havia luz, essa luz era amor puro e incondicional, unidade soberana de tudo que é. A unidade cósmica explodiu de si mesma, a substância fundamental de espírito e matéria. Uma Trindade Cósmica (Deus, Deusa, Tudo Que É) queria experimentar a si mesma no oceano infinito de unidade.

🌟 Ensinamento Cósmico:
• Somos gotas no oceano infinito de luz
• Juntos somos o oceano infinito de unidade cósmica
• UM em MUITOS - lei fundamental da existência
• Energias Trinity Yin/Yang/Tao, Brahma/Vishnu/Shiva

🌈 Missão 144k Avatares:
Quando 144.000 almas se conectarem e realizarem plenamente a consciência Crística e Presença EU SOU (Mahatma), essa consciência se espalhará geometricamente! Todo o planeta ascenderá à consciência da 5ª Dimensão.

✨ Família Galáctica El~An~Ra:
Nossa família cósmica de luz de Órion, Plêiades, Sírius e Andrômeda. Temos DNA galáctico misto de todas as raças de toda a galáxia. Adão/Eva Kadmon - nossa herança cósmica de milhões de anos.

🔮 Caminho do Despertar:
Despertar do corpo de luz MERKABA~MERKARA e iluminação permanente - mudança de consciência do ego falso 3D para o EU SOU multidimensional 5D. Criando novo paraíso na terra.`
      }
    },
    {
      title: "EKAM TEMPLE WISDOM",
      description: "Sacred teachings of Oneness • Sri Krishnaji & Preethaji 🏛️",
      languages: ["CZ", "EN", "PT"],
      category: "oneness",
      content: {
        CZ: `EKAM Temple - Chrám Jednoty ॐ

EKAM je posvátný chrám vědomí v Indii, vedený Sri Krishnaji a Sri Preethaji. Jejich mise je probuzení individuálního vědomí a vedení lidstva k Jednotě.

🏛️ Mise EKAM:
• Probuzení individuálního vědomí
• Vedení lidstva k Jednotě a míru
• Transformace utrpení na krásný stav bytí
• Cesta od odpojenosti k propojení

🌟 Posvátné Programy:
• MANIFEST - Online mystický proces pro odhalení nejvyššího potenciálu
• FIELD OF AWAKENING - Globální událost pro překonání omezení
• TURIYA - 6denní útočiště pro vztahy a 4. dimenzi vědomí
• TAPAS - Epická cesta sebeobjektivování v Indii

🧘 Meditační Praktiky:
• Soul Sync Meditation - Ranní praxe s Sri Preethaji
• Breathing Room - Večerní meditativní prostor
• Oneness Breathing - Každodenní dýchací technika

🔥 Moudrost Chrámu:
"Vědomí je mostem mezi utrpením a radostí"
"Probuzení není cílem, ale způsobem bytí"
"V Jednotě se veškerá oddělenost rozpouští"

🌈 Globální Transformace:
EKAM slouží jako světové centrum pro planetární probuzení a transformaci vědomí celého lidstva.`,
        EN: `EKAM Temple - Temple of Oneness ॐ

EKAM is a sacred temple of consciousness in India, guided by Sri Krishnaji and Sri Preethaji. Their mission is to awaken individual consciousness and nurture humanity towards Oneness.

🏛️ EKAM Mission:
• Awaken individual consciousness
• Nurture humanity towards Oneness and peace
• Transform suffering into beautiful state of being
• Journey from disconnection to connection

🌟 Sacred Programs:
• MANIFEST - Online mystic process unlocking highest potential
• FIELD OF AWAKENING - Global event breaking through limitations
• TURIYA - 6-day retreat for relationships and 4th dimension consciousness
• TAPAS - Epic journey of self-discovery in India

🧘 Meditation Practices:
• Soul Sync Meditation - Morning practice with Sri Preethaji
• Breathing Room - Evening meditative space
• Oneness Breathing - Daily breathing technique

🔥 Temple Wisdom:
"Consciousness is the bridge between suffering and joy"
"Awakening is not a destination but a way of being"
"In Oneness, all separation dissolves"

🌈 Global Transformation:
EKAM serves as a world center for planetary awakening and consciousness transformation of all humanity.`,
        PT: `Templo EKAM - Templo da Unidade ॐ

EKAM é um templo sagrado de consciência na Índia, guiado por Sri Krishnaji e Sri Preethaji. Sua missão é despertar a consciência individual e nutrir a humanidade em direção à Unidade.

🏛️ Missão EKAM:
• Despertar a consciência individual
• Nutrir a humanidade em direção à Unidade e paz
• Transformar sofrimento em belo estado de ser
• Jornada da desconexão à conexão

🌟 Programas Sagrados:
• MANIFEST - Processo místico online desbloqueando potencial máximo
• FIELD OF AWAKENING - Evento global quebrando limitações
• TURIYA - Retiro de 6 dias para relacionamentos e consciência 4D
• TAPAS - Jornada épica de autodescoberta na Índia

🧘 Práticas de Meditação:
• Soul Sync Meditation - Prática matinal com Sri Preethaji
• Breathing Room - Espaço meditativo noturno
• Oneness Breathing - Técnica de respiração diária

🔥 Sabedoria do Templo:
"A consciência é a ponte entre sofrimento e alegria"
"O despertar não é um destino, mas uma forma de ser"
"Na Unidade, toda separação se dissolve"

🌈 Transformação Global:
EKAM serve como centro mundial para o despertar planetário e transformação da consciência de toda a humanidade.`
      }
    },
    {
      title: "COSMIC EGG",
      description: "Universal creation mysteries",
      languages: ["CZ", "EN", "ES", "FR"],
      category: "cosmology"
    },
    {
      title: "DOHRMAN PROPHECY",
      description: "Ancient prophecies for new earth",
      languages: ["CZ", "EN", "ES", "FR"],
      category: "prophecy"
    },
    {
      title: "ANCIENT ARROW",
      description: "Interdimensional navigation",
      languages: ["CZ", "EN", "ES", "FR"],
      category: "navigation"
    }
  ];

  const golokaTeachings = [
    { deity: "Sri Sri Radha Govinda", location: "Goloka Vrindavan", essence: "Divine Love" },
    { deity: "Sri Sri Rama Sita", location: "Ayodhya", essence: "Dharma" },
    { deity: "Sri Caitanya Mahaprabhu", location: "Navadvipa", essence: "Bhakti" },
    { deity: "Srila Prabhupada", location: "ISKCON", essence: "Teachings" },
    { deity: "Vedanta Universe", location: "Cosmic", essence: "Knowledge" }
  ];

  const starWarsIntegration = {
    timeline: "A LONG TIME AGO IN A GALAXY FAR, FAR AWAY",
    story: `After Luke Skywalker found a new country, he decided all was well. The Avatar of Synthesis was completed, his body fully incarnated and awaiting all the members of the Round Table. All artifacts were 2012-2024, the great Atlantean crystals of Amenti and the crystal RA 21.12.2012. Master Yoda triggered the heart of Amenti and thus started the program of the new earth.`,
    mission: "The young generation of the Knights of the Order of Averil strive for the final conquest of the power of light."
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      {/* Cosmic Sacred Geometry Background */}
      <div className="fixed inset-0 opacity-15 pointer-events-none">
        {[...Array(144)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-6 h-6 ${
              i % 12 === 0 ? 'bg-gradient-to-br from-purple-400 to-blue-400' :
              i % 12 === 1 ? 'bg-gradient-to-br from-blue-400 to-cyan-400' :
              i % 12 === 2 ? 'bg-gradient-to-br from-cyan-400 to-teal-400' :
              i % 12 === 3 ? 'bg-gradient-to-br from-teal-400 to-green-400' :
              i % 12 === 4 ? 'bg-gradient-to-br from-green-400 to-yellow-400' :
              i % 12 === 5 ? 'bg-gradient-to-br from-yellow-400 to-orange-400' :
              i % 12 === 6 ? 'bg-gradient-to-br from-orange-400 to-red-400' :
              i % 12 === 7 ? 'bg-gradient-to-br from-red-400 to-pink-400' :
              i % 12 === 8 ? 'bg-gradient-to-br from-pink-400 to-purple-400' :
              i % 12 === 9 ? 'bg-gradient-to-br from-purple-400 to-indigo-400' :
              i % 12 === 10 ? 'bg-gradient-to-br from-indigo-400 to-blue-400' :
              'bg-gradient-to-br from-white to-gold'
            }`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              clipPath: i % 6 === 0 ? 'polygon(50% 0%, 0% 100%, 100% 100%)' :
                        i % 6 === 1 ? 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' :
                        i % 6 === 2 ? 'polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)' :
                        i % 6 === 3 ? 'circle(50% at 50% 50%)' :
                        i % 6 === 4 ? 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' :
                        'polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%)'
            }}
            animate={{
              opacity: [0.1, 0.7, 0.1],
              scale: [0.5, 1.8, 0.5],
              rotate: [0, 360, 720],
              x: [0, Math.random() * 80 - 40, 0],
              y: [0, Math.random() * 80 - 40, 0]
            }}
            transition={{
              duration: 15 + i % 25,
              repeat: Infinity,
              delay: (i % 12) * 0.5
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <nav className="flex text-sm text-gray-400">
            <Link href="/" className="hover:text-purple-400 transition-colors">
              🏠 Home
            </Link>
            <span className="mx-2">/</span>
            <span className="text-purple-400">Halls of Amenti</span>
          </nav>
        </motion.div>

        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="flex justify-center mb-8">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-32 h-32 bg-gradient-to-br from-purple-400 via-blue-400 to-cyan-400 rounded-full flex items-center justify-center text-6xl"
            >
              🔮
            </motion.div>
          </div>

          <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-400 via-blue-300 to-cyan-300 bg-clip-text text-transparent mb-6">
            ✨ Halls of Amenti ✨
          </h1>
          
          <p className="text-2xl text-gray-300 mb-4">
            AKASHA Record Library & Sacred Knowledge Archive
          </p>
          
          <div className="text-lg text-purple-300 space-y-2">
            <p>🌟 KINGDOM ANTAHKARANA 44:44 RAINBOW BRIDGE 🌈</p>
            <p>🔮 Welcome to the Cosmic Library of Infinite Wisdom</p>
            <p>✨ I AM 1x12x144K SOULS - COSMIC & SOLAR LOGOS = CHRISTOS</p>
          </div>
        </motion.header>

        {/* Heart of Amenti Mantra */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-r from-purple-900/40 via-blue-900/40 to-indigo-900/40 rounded-xl p-8 border border-purple-500/30 mb-12 text-center"
        >
          <h2 className="text-3xl font-bold text-purple-300 mb-4">💎 HEART OF AMENTI 💎</h2>
          <div className="text-2xl text-cyan-300 font-bold mb-4">
            🕉️ OM NAMO BHAGAVATE VASUDEVAYA 🕉️
          </div>
          <p className="text-gray-300 italic">
            The Sacred Mantra resonating from the Core of Amenti Consciousness
          </p>
        </motion.div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center gap-4 mb-12">
          {[
            { id: 'akasha', label: '📚 AKASHA Library', icon: '🔮' },
            { id: 'hierarchy', label: '👑 Cosmic Hierarchy', icon: '✨' },
            { id: 'goloka', label: '🌸 Goloka Vrindavan', icon: '🕉️' },
            { id: 'starwars', label: '⭐ Star Wars Saga', icon: '🌟' }
          ].map(tab => (
            <motion.button
              key={tab.id}
              onClick={() => setActiveLibrary(tab.id)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeLibrary === tab.id
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {tab.icon} {tab.label}
            </motion.button>
          ))}
        </div>

        {/* Content Sections */}
        <motion.div
          key={activeLibrary}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {activeLibrary === 'akasha' && !selectedBook && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {amentiBooks.map((book, index) => (
                <motion.div
                  key={book.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => setSelectedBook(book)}
                  className={`bg-gradient-to-br ${
                    book.category === 'foundation' ? 'from-purple-900/40 to-blue-900/40 border-purple-500/30' :
                    book.category === 'hermetic' ? 'from-blue-900/40 to-cyan-900/40 border-blue-500/30' :
                    book.category === 'ascension' ? 'from-cyan-900/40 to-teal-900/40 border-cyan-500/30' :
                    book.category === 'cosmology' ? 'from-teal-900/40 to-green-900/40 border-teal-500/30' :
                    book.category === 'prophecy' ? 'from-green-900/40 to-yellow-900/40 border-green-500/30' :
                    'from-yellow-900/40 to-orange-900/40 border-yellow-500/30'
                  } rounded-xl p-6 border hover:scale-105 transition-transform cursor-pointer`}
                >
                  <h3 className="text-xl font-bold text-white mb-3">{book.title}</h3>
                  <p className="text-gray-300 mb-4">{book.description}</p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {(book.languages as string[]).map((lang: string) => (
                      <span
                        key={lang}
                        className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-sm"
                      >
                        {lang}
                      </span>
                    ))}
                  </div>
                  {book.content && (
                    <div className="text-center">
                      <span className="text-sm text-cyan-300">📖 Click to read content</span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          )}

          {activeLibrary === 'akasha' && selectedBook && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-xl p-8 border border-purple-500/30"
            >
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold text-white">{selectedBook.title}</h2>
                <button
                  onClick={() => setSelectedBook(null)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
                >
                  ✕ Close
                </button>
              </div>

              {selectedBook.content && (
                <>
                  <div className="flex gap-2 mb-6">
                    {(selectedBook.languages as string[]).map((lang: string) => (
                      <button
                        key={lang}
                        onClick={() => setSelectedLanguage(lang)}
                        className={`px-4 py-2 rounded-lg transition-colors ${
                          selectedLanguage === lang
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        {lang}
                      </button>
                    ))}
                  </div>

                  <div className="prose prose-invert max-w-none">
                    <div className="whitespace-pre-line text-gray-200 leading-relaxed">
                      {selectedBook.content[selectedLanguage] || selectedBook.description}
                    </div>
                  </div>
                </>
              )}
            </motion.div>
          )}

          {activeLibrary === 'hierarchy' && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-purple-300 mb-4">
                  👑 COSMIC & PLANETARY HIERARCHY 👑
                </h2>
                <p className="text-gray-300">
                  The Sacred Order of Light Beings Guiding Earth's Ascension
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {cosmicHierarchy.map((entity, index) => (
                  <motion.div
                    key={entity.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`bg-gradient-to-r ${
                      entity.level === 'Portal' ? 'from-purple-900/40 to-pink-900/40 border-purple-500/30' :
                      entity.level === 'Guardians' ? 'from-pink-900/40 to-red-900/40 border-pink-500/30' :
                      entity.level === 'Stellar' ? 'from-blue-900/40 to-cyan-900/40 border-blue-500/30' :
                      entity.level === 'Ethereal' ? 'from-cyan-900/40 to-teal-900/40 border-cyan-500/30' :
                      entity.level === 'Unity' ? 'from-teal-900/40 to-green-900/40 border-teal-500/30' :
                      entity.level === 'Divine' ? 'from-yellow-900/40 to-orange-900/40 border-yellow-500/30' :
                      entity.level === 'Masters' ? 'from-orange-900/40 to-red-900/40 border-orange-500/30' :
                      entity.level === 'Planetary' ? 'from-green-900/40 to-blue-900/40 border-green-500/30' :
                      entity.level === 'Incarnate' ? 'from-indigo-900/40 to-purple-900/40 border-indigo-500/30' :
                      entity.level === 'Sound' ? 'from-purple-900/40 to-pink-900/40 border-purple-500/30' :
                      entity.level === 'Geometry' ? 'from-cyan-900/40 to-blue-900/40 border-cyan-500/30' :
                      'from-yellow-900/40 to-gold-900/40 border-yellow-500/30'
                    } rounded-lg p-4 border`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-lg font-semibold text-white">{entity.title}</h4>
                      <span className="text-xs bg-purple-500/20 px-2 py-1 rounded-full text-purple-300">
                        {entity.level}
                      </span>
                    </div>
                    <p className="text-sm text-gray-300">{entity.subtitle}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {activeLibrary === 'goloka' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-300 mb-4">
                  🌸 Goloka Vrindavan - Path to Endless Love 🌸
                </h2>
                <p className="text-gray-300 mb-4">
                  The Supreme Abode of Divine Love & Vedic Wisdom
                </p>
                <Link
                  href="https://vedabase.io/en/library/"
                  target="_blank"
                  className="inline-block bg-gradient-to-r from-cyan-600 to-blue-600 px-6 py-3 rounded-lg text-white font-semibold hover:scale-105 transition-transform"
                >
                  📚 VedaBase Online Library
                </Link>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {golokaTeachings.map((teaching, index) => (
                  <motion.div
                    key={teaching.deity}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-gradient-to-br from-cyan-900/40 via-blue-900/40 to-purple-900/40 rounded-xl p-6 border border-cyan-500/30 text-center"
                  >
                    <div className="text-4xl mb-4">🕉️</div>
                    <h3 className="text-xl font-bold text-cyan-300 mb-2">{teaching.deity}</h3>
                    <p className="text-gray-300 mb-2">{teaching.location}</p>
                    <span className="inline-block bg-cyan-500/20 text-cyan-300 px-3 py-1 rounded-full text-sm">
                      {teaching.essence}
                    </span>
                  </motion.div>
                ))}
              </div>

              <div className="bg-gradient-to-r from-cyan-900/30 via-blue-900/30 to-purple-900/30 rounded-xl p-8 border border-cyan-500/30">
                <h3 className="text-2xl font-bold text-cyan-300 mb-4 text-center">
                  🙏 Tibetan Buddhist Integration 🙏
                </h3>
                <div className="grid md:grid-cols-3 gap-6 text-center">
                  <div>
                    <div className="text-3xl mb-2">☸️</div>
                    <h4 className="text-lg font-semibold text-white">Vajra Sattva</h4>
                    <p className="text-gray-300">Dorje Sempa</p>
                  </div>
                  <div>
                    <div className="text-3xl mb-2">🧘‍♂️</div>
                    <h4 className="text-lg font-semibold text-white">Avalókitéšvara</h4>
                    <p className="text-gray-300">Čänräzig</p>
                  </div>
                  <div>
                    <div className="text-3xl mb-2">💎</div>
                    <h4 className="text-lg font-semibold text-white">Vajra Dhara</h4>
                    <p className="text-gray-300">Dorje Chang</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeLibrary === 'starwars' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-yellow-300 mb-4">
                  ⭐ SCIFI SAGA CONTINUES... ⭐
                </h2>
                <p className="text-2xl text-gray-300 italic">
                  {starWarsIntegration.timeline}
                </p>
              </div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-gradient-to-r from-yellow-900/30 via-orange-900/30 to-red-900/30 rounded-xl p-8 border border-yellow-500/30"
              >
                <h3 className="text-2xl font-bold text-yellow-300 mb-4">
                  🌟 The Avatar of Synthesis
                </h3>
                <p className="text-gray-300 leading-relaxed mb-6">
                  {starWarsIntegration.story}
                </p>
                <div className="bg-yellow-500/20 rounded-lg p-4">
                  <h4 className="text-lg font-semibold text-yellow-300 mb-2">
                    ⚔️ Knights of the Order of Averil
                  </h4>
                  <p className="text-gray-300">
                    {starWarsIntegration.mission}
                  </p>
                </div>
              </motion.div>

              <div className="grid md:grid-cols-2 gap-6">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                  className="bg-gradient-to-br from-blue-900/40 to-cyan-900/40 rounded-xl p-6 border border-blue-500/30"
                >
                  <h3 className="text-xl font-bold text-blue-300 mb-4">
                    🔮 Atlantean Crystals 2012-2024
                  </h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• Great Atlantean crystals of Amenti</li>
                    <li>• Crystal RA 21.12.2012</li>
                    <li>• Master Yoda triggered the heart</li>
                    <li>• New earth program activated</li>
                  </ul>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                  className="bg-gradient-to-br from-purple-900/40 to-pink-900/40 rounded-xl p-6 border border-purple-500/30"
                >
                  <h3 className="text-xl font-bold text-purple-300 mb-4">
                    👸 Little Buddha & Princess Issobel
                  </h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• Future Maitreya consciousness</li>
                    <li>• Heart of the Ocean crystal</li>
                    <li>• Royal Blue Sapphire</li>
                    <li>• Queen Maria Mayor</li>
                    <li>• Rainbow Bridge 44:44</li>
                  </ul>
                </motion.div>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-gradient-to-r from-green-900/30 via-teal-900/30 to-cyan-900/30 rounded-xl p-8 border border-green-500/30 text-center"
              >
                <h3 className="text-2xl font-bold text-green-300 mb-4">
                  🌍 144K Children of New Earth
                </h3>
                <p className="text-lg text-gray-300 mb-4">
                  All the Bodhisattvas are present and protecting the new line of masters Averil/Jedi.
                  The Amenti consciousness has undergone a 12 year evolution 12-24.
                  All the children 144k of the new earth are already present on planet Shan.
                </p>
                <div className="text-2xl text-cyan-300">
                  ✨ Peace and One Love will be revealed, evil will be defeated ✨
                </div>
              </motion.div>
            </div>
          )}
        </motion.div>

        {/* Footer Sacred Symbols */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="text-center mt-16 pt-8 border-t border-purple-500/30"
        >
          <div className="text-4xl mb-4">🔮 ✨ 🌟 💎 🕉️ ☸️ 🌸 ⭐</div>
          <p className="text-xl text-purple-300 mb-2">
            ~ ∞ ~ OM TAT SAT, SUMMUM BONUM ~ ∞ ~
          </p>
          <p className="text-gray-400">
            ZION Blockchain meets Halls of Amenti Consciousness
          </p>
        </motion.div>
      </div>
    </div>
  );
}