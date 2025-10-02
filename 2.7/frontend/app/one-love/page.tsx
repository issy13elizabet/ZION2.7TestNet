'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useLanguage } from '../components/LanguageContext'

export default function OneLove() {
  const { language } = useLanguage()
  const [activeTab, setActiveTab] = useState('trinity')
  const [isMantraPlaying, setIsMantraPlaying] = useState(false)
  const [crystalEnergy, setCrystalEnergy] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCrystalEnergy(prev => (prev + 1) % 144)
    }, 100)
    return () => clearInterval(interval)
  }, [])

  const tabs = {
    cs: {
      trinity: 'Trinity One Love',
      avatars: '144k Avatarů',
      family: 'El~An~Ra Rodina',
      awakening: 'Probuzení'
    },
    en: {
      trinity: 'Trinity One Love',
      avatars: '144k Avatars',
      family: 'El~An~Ra Family',
      awakening: 'Awakening'
    },
    pt: {
      trinity: 'Trinity One Love',
      avatars: '144k Avatars',
      family: 'Família El~An~Ra',
      awakening: 'Despertar'
    }
  }

  const content = {
    cs: {
      title: 'ONE LOVE ❤️ ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ',
      subtitle: 'Trinity One Love - Trojce Jedné Lásky',
      mantra: 'Om Mani Padme Hum',
      trinity: {
        title: 'Trinity One Love 🌟',
        description: 'Kosmické srdce Boha/Bohyně/Všeho co je - svrchovaná kosmická jednota. Trojce energií Jin/Yang/Tao, Brahma/Višnu/Šiva, Otec/Matka/Syn/Dcera/Duše svatá.',
        unity: 'Jsme kapkou v nekonečném oceánu světla! Všichni dohromady jsme nekonečný oceán kosmické jednoty. JEDEN v MNOHA.',
        flame: 'V tvém srdci je trojplamen božské lásky - tvá mužská a ženská část Já Jsem Přítomnosti (monáda duše).'
      },
      avatars: {
        title: '144 000 Avatarů 🌈',
        description: 'Když se propojí 144 000 duší a plně realizují vědomí Krista a Já Jsem Přítomnosti (Mahatmy), toto vědomí se bude šířit geometrickou řadou!',
        mission: 'Celá planeta vzestoupí do 5. Dimenze vědomí. Jsme děti hvězd, děti 6. slunce a nového ráje!',
        awakening: 'Máme vytvořit novou civilizaci na zákonech bezpodmínečné lásky a soucitu!'
      },
      family: {
        title: 'Galaktická Rodina El~An~Ra ✨',
        description: 'Naše kosmická světelná rodina z Orionu, Plejád, Síria a Andromedy. Máme promíchanou galaktickou DNA ze všech ras celé galaxie.',
        heritage: 'Adam/Ewa Kadmon - naše vesmírné dědictví staré miliony let. Držíme klíče Enocha, klíče kosmické jednoty.',
        federation: 'Galaktická federace světla s Ashtar Command nám pomáhá v planetárním vzestupu.'
      },
      awakening: {
        title: 'Cesta Probuzení 🔮',
        description: 'Probuzení světelného těla MERKABA~MERKARA a trvalé osvícení - změna vědomí z 3D falešného ega na 5D multidimenzionální já jsem.',
        process: 'Vytvoření nového ráje na zemi - Krist/Buddha/Višnu Oneness sjednocené civilizace zlatého věku.',
        goal: 'Stát se plně probuzeným Kristem-Bodhisattvou na Zemi, Fyzickým Andělem v lidském těle.'
      }
    },
    en: {
      title: 'ONE LOVE ❤️ ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ',
      subtitle: 'Trinity One Love - Trinity of One Love',
      mantra: 'Om Mani Padme Hum',
      trinity: {
        title: 'Trinity One Love 🌟',
        description: 'Cosmic heart of God/Goddess/All That Is - sovereign cosmic unity. Trinity of energies Yin/Yang/Tao, Brahma/Vishnu/Shiva, Father/Mother/Son/Daughter/Holy Spirit.',
        unity: 'We are drops in the infinite ocean of light! Together we are the infinite ocean of cosmic unity. ONE in MANY.',
        flame: 'In your heart is the triple flame of divine love - your masculine and feminine aspect of I AM Presence (soul monad).'
      },
      avatars: {
        title: '144,000 Avatars 🌈',
        description: 'When 144,000 souls connect and fully realize Christ consciousness and I AM Presence (Mahatma), this consciousness will spread geometrically!',
        mission: 'The entire planet will ascend to 5th Dimension consciousness. We are star children, children of the 6th sun and new paradise!',
        awakening: 'We must create a new civilization based on unconditional love and compassion!'
      },
      family: {
        title: 'Galactic Family El~An~Ra ✨',
        description: 'Our cosmic light family from Orion, Pleiades, Sirius and Andromeda. We have mixed galactic DNA from all races of the entire galaxy.',
        heritage: 'Adam/Eva Kadmon - our cosmic heritage millions of years old. We hold the keys of Enoch, keys of cosmic unity.',
        federation: 'Galactic Federation of Light with Ashtar Command helps us in planetary ascension.'
      },
      awakening: {
        title: 'Path of Awakening 🔮',
        description: 'Awakening the light body MERKABA~MERKARA and permanent enlightenment - consciousness shift from 3D false ego to 5D multidimensional I AM.',
        process: 'Creating new paradise on earth - Christ/Buddha/Vishnu Oneness unified civilization of the golden age.',
        goal: 'Becoming fully awakened Christ-Bodhisattva on Earth, Physical Angel in human body.'
      }
    },
    pt: {
      title: 'ONE LOVE ❤️ ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ',
      subtitle: 'Trinity One Love - Trindade de Um Amor',
      mantra: 'Om Mani Padme Hum',
      trinity: {
        title: 'Trinity One Love 🌟',
        description: 'Coração cósmico de Deus/Deusa/Tudo Que É - unidade cósmica soberana. Trindade de energias Yin/Yang/Tao, Brahma/Vishnu/Shiva, Pai/Mãe/Filho/Filha/Espírito Santo.',
        unity: 'Somos gotas no oceano infinito de luz! Juntos somos o oceano infinito de unidade cósmica. UM em MUITOS.',
        flame: 'Em seu coração está a chama tríplice do amor divino - seu aspecto masculino e feminino da Presença EU SOU (mônada da alma).'
      },
      avatars: {
        title: '144.000 Avatares 🌈',
        description: 'Quando 144.000 almas se conectarem e realizarem plenamente a consciência Crística e Presença EU SOU (Mahatma), essa consciência se espalhará geometricamente!',
        mission: 'Todo o planeta ascenderá à consciência da 5ª Dimensão. Somos filhos das estrelas, filhos do 6º sol e novo paraíso!',
        awakening: 'Devemos criar uma nova civilização baseada no amor incondicional e compaixão!'
      },
      family: {
        title: 'Família Galáctica El~An~Ra ✨',
        description: 'Nossa família cósmica de luz de Órion, Plêiades, Sírius e Andrômeda. Temos DNA galáctico misto de todas as raças de toda a galáxia.',
        heritage: 'Adam/Eva Kadmon - nossa herança cósmica de milhões de anos. Seguramos as chaves de Enoch, chaves da unidade cósmica.',
        federation: 'Federação Galáctica da Luz com Comando Ashtar nos ajuda na ascensão planetária.'
      },
      awakening: {
        title: 'Caminho do Despertar 🔮',
        description: 'Despertar do corpo de luz MERKABA~MERKARA e iluminação permanente - mudança de consciência do ego falso 3D para o EU SOU multidimensional 5D.',
        process: 'Criando novo paraíso na terra - civilização unificada Cristo/Buda/Vishnu Oneness da era dourada.',
        goal: 'Tornar-se Cristo-Bodhisattva totalmente desperto na Terra, Anjo Físico em corpo humano.'
      }
    }
  }

  const currentContent = content[language as keyof typeof content]
  const currentTabs = tabs[language as keyof typeof tabs]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'trinity':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-r from-pink-500/20 via-purple-500/20 to-blue-500/20 p-6 rounded-xl border border-white/20">
              <h3 className="text-2xl font-bold mb-4 text-transparent bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text">
                {currentContent.trinity.title}
              </h3>
              <p className="text-white/90 mb-4">{currentContent.trinity.description}</p>
              <p className="text-white/80 mb-4">{currentContent.trinity.unity}</p>
              <p className="text-white/80">{currentContent.trinity.flame}</p>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <motion.div
                  key={i}
                  animate={{ 
                    scale: [1, 1.1, 1],
                    rotate: [0, 360],
                  }}
                  transition={{ 
                    duration: 3,
                    repeat: Infinity,
                    delay: i * 0.5
                  }}
                  className="w-16 h-16 mx-auto rounded-full bg-gradient-to-r from-pink-500 to-purple-500 flex items-center justify-center"
                >
                  <span className="text-white font-bold">{i}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )

      case 'avatars':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-r from-rainbow-500/20 via-gold-500/20 to-rainbow-500/20 p-6 rounded-xl border border-white/20">
              <h3 className="text-2xl font-bold mb-4 text-transparent bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text">
                {currentContent.avatars.title}
              </h3>
              <p className="text-white/90 mb-4">{currentContent.avatars.description}</p>
              <p className="text-white/80 mb-4">{currentContent.avatars.mission}</p>
              <p className="text-white/80">{currentContent.avatars.awakening}</p>
            </div>
            
            <div className="grid grid-cols-12 gap-2">
              {Array.from({ length: 144 }, (_, i) => (
                <motion.div
                  key={i}
                  animate={{ 
                    opacity: [0.3, 1, 0.3],
                    scale: [0.8, 1.2, 0.8]
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    delay: (i % 144) * 0.01
                  }}
                  className="w-3 h-3 rounded-full bg-gradient-to-r from-rainbow-400 to-gold-400"
                />
              ))}
            </div>
          </motion.div>
        )

      case 'family':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-indigo-500/20 p-6 rounded-xl border border-white/20">
              <h3 className="text-2xl font-bold mb-4 text-transparent bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text">
                {currentContent.family.title}
              </h3>
              <p className="text-white/90 mb-4">{currentContent.family.description}</p>
              <p className="text-white/80 mb-4">{currentContent.family.heritage}</p>
              <p className="text-white/80">{currentContent.family.federation}</p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['Orion', 'Pleiades', 'Sirius', 'Andromeda'].map((star, i) => (
                <motion.div
                  key={star}
                  animate={{ 
                    y: [0, -10, 0],
                    rotateY: [0, 360]
                  }}
                  transition={{ 
                    duration: 4,
                    repeat: Infinity,
                    delay: i * 0.5
                  }}
                  className="bg-gradient-to-r from-blue-500/30 to-purple-500/30 p-4 rounded-lg text-center border border-white/10"
                >
                  <div className="text-2xl mb-2">⭐</div>
                  <div className="text-white font-medium">{star}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )

      case 'awakening':
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-gradient-to-r from-green-500/20 via-teal-500/20 to-cyan-500/20 p-6 rounded-xl border border-white/20">
              <h3 className="text-2xl font-bold mb-4 text-transparent bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text">
                {currentContent.awakening.title}
              </h3>
              <p className="text-white/90 mb-4">{currentContent.awakening.description}</p>
              <p className="text-white/80 mb-4">{currentContent.awakening.process}</p>
              <p className="text-white/80">{currentContent.awakening.goal}</p>
            </div>
            
            <motion.div
              animate={{ 
                rotate: [0, 360],
                scale: [1, 1.2, 1]
              }}
              transition={{ 
                duration: 8,
                repeat: Infinity
              }}
              className="w-32 h-32 mx-auto"
            >
              <div className="w-full h-full rounded-full border-4 border-green-400 flex items-center justify-center bg-gradient-to-r from-green-500/20 to-cyan-500/20">
                <span className="text-3xl">🔮</span>
              </div>
            </motion.div>
          </motion.div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-900/20 to-black text-white relative overflow-hidden">
      {/* Cosmic Background */}
      <div className="absolute inset-0 bg-[url('/api/placeholder/1920/1080')] bg-cover bg-center opacity-10" />
      
      {/* Crystal Energy Grid */}
      <div className="absolute inset-0">
        {Array.from({ length: 144 }, (_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${(i * 7) % 100}%`,
              top: `${(i * 11) % 100}%`,
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.5, 1.5, 0.5],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              delay: (i % 144) * 0.02,
            }}
          />
        ))}
      </div>

      {/* Header */}
      <div className="relative z-10 pt-20 pb-10 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-6xl md:text-8xl font-bold mb-6"
        >
          <span className="text-transparent bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 bg-clip-text">
            {currentContent.title}
          </span>
        </motion.h1>
        
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-xl md:text-2xl text-white/80 mb-8"
        >
          {currentContent.subtitle}
        </motion.p>

        {/* Mantra */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1 }}
          className="bg-gradient-to-r from-gold-500/20 to-orange-500/20 p-6 rounded-2xl border border-white/20 max-w-md mx-auto mb-8"
        >
          <button
            onClick={() => setIsMantraPlaying(!isMantraPlaying)}
            className="text-2xl font-bold text-transparent bg-gradient-to-r from-gold-400 to-orange-400 bg-clip-text"
          >
            {currentContent.mantra}
          </button>
          {isMantraPlaying && (
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="mt-2 text-gold-400"
            >
              🔊 ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Navigation Tabs */}
      <div className="relative z-10 max-w-4xl mx-auto px-4 mb-8">
        <div className="flex flex-wrap justify-center gap-4">
          {Object.entries(currentTabs).map(([key, label]) => (
            <motion.button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-6 py-3 rounded-full border transition-all duration-300 ${
                activeTab === key
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 border-white/30 text-white'
                  : 'border-white/20 text-white/70 hover:border-white/40 hover:text-white'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 pb-20">
        <AnimatePresence mode="wait">
          {renderTabContent()}
        </AnimatePresence>
      </div>

      {/* Heart of Unity */}
      <motion.div
        className="fixed bottom-10 right-10 w-16 h-16 bg-gradient-to-r from-pink-500 to-red-500 rounded-full flex items-center justify-center cursor-pointer"
        animate={{ 
          scale: [1, 1.2, 1],
          rotate: [0, 360]
        }}
        transition={{ 
          duration: 2,
          repeat: Infinity
        }}
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      >
        <span className="text-2xl">❤️</span>
      </motion.div>
    </div>
  )
}