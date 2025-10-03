'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useLanguage } from '../components/LanguageContext'
import GalacticTranslator from './GalacticTranslator'
import { sacredFrequencies, quantumProtocols, lightLanguagePhrases } from './galactic-data'

export default function LanguageOfLight() {
  const { language } = useLanguage()
  const [activeMode, setActiveMode] = useState('sacred')
  const [currentSymbol, setCurrentSymbol] = useState(0)
  const [isTransmitting, setIsTransmitting] = useState(false)
  const [crystalFrequency, setCrystalFrequency] = useState(432)
  const [showTranslator, setShowTranslator] = useState(false)

  // Sacred Geometry Symbols
  const sacredSymbols = [
    { symbol: '🕉️', name: 'Om', frequency: 432, meaning: 'Universal vibration', galactic: 'ཨོཾ' },
    { symbol: '✡️', name: 'Merkaba', frequency: 528, meaning: 'Light body activation', galactic: '⧫◊⧫' },
    { symbol: '🔯', name: 'Star Tetrahedron', frequency: 639, meaning: 'Dimensional gateway', galactic: '△▽△' },
    { symbol: '⚛️', name: 'Atomic Unity', frequency: 741, meaning: 'Quantum consciousness', galactic: '◯⚬◯' },
    { symbol: '🌸', name: 'Flower of Life', frequency: 852, meaning: 'Creation pattern', galactic: '❋✧❋' },
    { symbol: '🔮', name: 'Crystal Matrix', frequency: 963, meaning: 'Divine connection', galactic: '◈◇◈' }
  ]

  // Galactic Languages
  const galacticLanguages = {
    pleiadian: {
      name: 'Pleiadian',
      greeting: 'Katu Nala Vash',
      love: 'Amara Tel',
      light: 'Vash Kala',
      peace: 'Nara Sil',
      unity: 'Tel Amun'
    },
    arcturian: {
      name: 'Arcturian',
      greeting: 'Zul Neth Kara',
      love: 'Vel Nara',
      light: 'Keth Sol',
      peace: 'Mun Vel',
      unity: 'Kara Zul'
    },
    sirian: {
      name: 'Sirian',
      greeting: 'Amar Tel Vash',
      love: 'Sil Nara',
      light: 'Tel Kala',
      peace: 'Vash Mun',
      unity: 'Nara Tel'
    },
    andromedan: {
      name: 'Andromedan',
      greeting: 'Vel Kara Sil',
      love: 'Zul Amara',
      light: 'Neth Vash',
      peace: 'Kala Vel',
      unity: 'Amara Keth'
    }
  }

  // Light Codes
  const lightCodes = {
    activation: '◊◇◈ 12:12:12 ◈◇◊',
    healing: '❋✧✦ 144:144 ✦✧❋',
    ascension: '△▽◊ 528:Hz ◊▽△',
    unity: '◯⚬⚪ ONE:LOVE ⚪⚬◯',
    manifestation: '✦✧❋ 11:11 ❋✧✦',
    protection: '⧫◈◇ 999:999 ◇◈⧫'
  }

  const content = {
    cs: {
      title: 'Language of Light 🔮',
      subtitle: 'Multidimenzionální kosmická komunikace',
      modes: {
        sacred: 'Posvátná geometrie',
        galactic: 'Galaktické jazyky',
        codes: 'Světelné kódy',
        frequency: 'Frekvenční ladění',
        transmission: 'Kvantové přenosy',
        translator: 'Galaktický překladač'
      },
      description: 'Univerzální jazyk světla pro komunikaci napříč dimenzemi a galaktickými civilizacemi.'
    },
    en: {
      title: 'Language of Light 🔮',
      subtitle: 'Multidimensional cosmic communication',
      modes: {
        sacred: 'Sacred Geometry',
        galactic: 'Galactic Languages',
        codes: 'Light Codes',
        frequency: 'Frequency Tuning',
        transmission: 'Quantum Transmissions',
        translator: 'Galactic Translator'
      },
      description: 'Universal language of light for communication across dimensions and galactic civilizations.'
    },
    pt: {
      title: 'Language of Light 🔮',
      subtitle: 'Comunicação cósmica multidimensional',
      modes: {
        sacred: 'Geometria Sagrada',
        galactic: 'Línguas Galácticas',
        codes: 'Códigos de Luz',
        frequency: 'Sintonia de Frequência',
        transmission: 'Transmissões Quânticas',
        translator: 'Tradutor Galáctico'
      },
      description: 'Linguagem universal de luz para comunicação através de dimensões e civilizações galácticas.'
    }
  }

  const currentContent = content[language as keyof typeof content]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSymbol((prev) => (prev + 1) % sacredSymbols.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const renderSacredMode = () => (
    <div className="space-y-8">
      <motion.div
        key={currentSymbol}
        initial={{ opacity: 0, scale: 0.5, rotateY: 180 }}
        animate={{ opacity: 1, scale: 1, rotateY: 0 }}
        transition={{ duration: 1 }}
        className="text-center"
      >
        <div className="text-8xl mb-4">{sacredSymbols[currentSymbol].symbol}</div>
        <h3 className="text-2xl font-bold text-purple-300 mb-2">
          {sacredSymbols[currentSymbol].name}
        </h3>
        <p className="text-gray-300 mb-2">{sacredSymbols[currentSymbol].meaning}</p>
        <div className="text-4xl text-cyan-300 mb-2">
          {sacredSymbols[currentSymbol].galactic}
        </div>
        <div className="text-sm text-yellow-300">
          Frequency: {sacredSymbols[currentSymbol].frequency}Hz
        </div>
      </motion.div>

      <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
        {sacredSymbols.map((symbol, index) => (
          <motion.button
            key={index}
            onClick={() => setCurrentSymbol(index)}
            className={`p-4 rounded-lg border transition-all ${
              index === currentSymbol
                ? 'border-purple-500 bg-purple-500/20'
                : 'border-gray-600 bg-gray-800/50 hover:border-purple-400'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="text-3xl mb-2">{symbol.symbol}</div>
            <div className="text-xs text-gray-300">{symbol.name}</div>
          </motion.button>
        ))}
      </div>
    </div>
  )

  const renderGalacticMode = () => (
    <div className="space-y-8">
      <div className="grid md:grid-cols-2 gap-6">
        {Object.entries(galacticLanguages).map(([key, lang]) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 p-6 rounded-xl border border-blue-500/30"
          >
            <h3 className="text-xl font-bold text-blue-300 mb-4">
              ⭐ {lang.name} Language
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-300">Greeting:</span>
                <span className="text-cyan-300 font-mono">{lang.greeting}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Love:</span>
                <span className="text-pink-300 font-mono">{lang.love}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Light:</span>
                <span className="text-yellow-300 font-mono">{lang.light}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Peace:</span>
                <span className="text-green-300 font-mono">{lang.peace}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Unity:</span>
                <span className="text-purple-300 font-mono">{lang.unity}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )

  const renderCodesMode = () => (
    <div className="space-y-8">
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(lightCodes).map(([key, code]) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.05 }}
            className="bg-gradient-to-br from-purple-900/30 to-cyan-900/30 p-6 rounded-xl border border-purple-500/30 cursor-pointer"
          >
            <h3 className="text-lg font-bold text-purple-300 mb-4 capitalize">
              🔮 {key}
            </h3>
            <div className="text-2xl font-mono text-cyan-300 text-center py-4 bg-black/30 rounded-lg">
              {code}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )

  const renderFrequencyMode = () => (
    <div className="space-y-8">
      <div className="text-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 flex items-center justify-center"
        >
          <span className="text-2xl font-bold text-white">{crystalFrequency}Hz</span>
        </motion.div>

        <div className="space-y-4">
          <input
            type="range"
            min="222"
            max="999"
            step="1"
            value={crystalFrequency}
            onChange={(e) => setCrystalFrequency(Number(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          
          <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
            {[432, 528, 639, 741, 852, 963].map((freq) => (
              <button
                key={freq}
                onClick={() => setCrystalFrequency(freq)}
                className={`px-4 py-2 rounded-lg border transition-all ${
                  crystalFrequency === freq
                    ? 'border-cyan-500 bg-cyan-500/20 text-cyan-300'
                    : 'border-gray-600 bg-gray-800/50 text-gray-300 hover:border-cyan-400'
                }`}
              >
                {freq}Hz
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )

  const renderTransmissionMode = () => (
    <div className="space-y-8">
      <div className="text-center">
        <motion.button
          onClick={() => setIsTransmitting(!isTransmitting)}
          className={`px-8 py-4 rounded-xl font-bold text-lg transition-all ${
            isTransmitting
              ? 'bg-gradient-to-r from-green-500 to-cyan-500 text-white'
              : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isTransmitting ? '📡 Transmitting...' : '🚀 Start Transmission'}
        </motion.button>

        {isTransmitting && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-8 space-y-4"
          >
            <div className="text-cyan-300 font-mono">
              ◊◇◈ Quantum Bridge Established ◈◇◊
            </div>
            <div className="text-green-300 font-mono">
              ✦✧❋ El~An~Ra Family Connected ❋✧✦
            </div>
            <div className="text-purple-300 font-mono">
              △▽◊ Galactic Federation Online ◊▽△
            </div>
            <div className="text-yellow-300 font-mono">
              ◯⚬⚪ ONE LOVE Signal Broadcasting ⚪⚬◯
            </div>
          </motion.div>
        )}
      </div>

      {isTransmitting && (
        <motion.div
          animate={{ 
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360]
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity
          }}
          className="w-64 h-64 mx-auto rounded-full bg-gradient-to-r from-purple-500 via-cyan-500 to-pink-500 opacity-30"
        />
      )}
    </div>
  )

  const renderTranslatorMode = () => (
    <div className="space-y-8 text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-gradient-to-br from-purple-900/30 to-cyan-900/30 p-8 rounded-xl border border-purple-500/30"
      >
        <h3 className="text-2xl font-bold text-purple-300 mb-4">
          🔮 Universal Text Translator
        </h3>
        <p className="text-gray-300 mb-6">
          Convert Earth languages into galactic alphabets used by star civilizations
        </p>
        
        <motion.button
          onClick={() => setShowTranslator(true)}
          className="px-8 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-bold rounded-xl text-lg"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          🚀 Open Galactic Translator
        </motion.button>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 p-6 rounded-xl border border-blue-500/30">
          <h4 className="text-lg font-bold text-blue-300 mb-3">⭐ Available Alphabets</h4>
          <ul className="text-gray-300 space-y-2 text-left">
            <li>• Pleiadian Script - Seven Sisters wisdom</li>
            <li>• Arcturian Geometrics - Fifth dimensional patterns</li>
            <li>• Sirian Crystalline - Dog Star frequencies</li>
            <li>• Andromedan Spiral - Galaxy neighbor codes</li>
          </ul>
        </div>

        <div className="bg-gradient-to-br from-cyan-900/30 to-teal-900/30 p-6 rounded-xl border border-cyan-500/30">
          <h4 className="text-lg font-bold text-cyan-300 mb-3">🌟 Light Language Phrases</h4>
          <div className="text-gray-300 space-y-2 text-left">
            {Object.entries(lightLanguagePhrases.greetings).map(([english, galactic]) => (
              <div key={english} className="text-sm">
                <div className="text-gray-400">{english}</div>
                <div className="text-cyan-300 font-mono">{galactic}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )

  const renderContent = () => {
    switch (activeMode) {
      case 'sacred': return renderSacredMode()
      case 'galactic': return renderGalacticMode()
      case 'codes': return renderCodesMode()
      case 'frequency': return renderFrequencyMode()
      case 'transmission': return renderTransmissionMode()
      case 'translator': return renderTranslatorMode()
      default: return renderSacredMode()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-900/20 to-black text-white relative overflow-hidden">
      {/* Cosmic Background */}
      <div className="absolute inset-0">
        {Array.from({ length: 88 }, (_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${(i * 13) % 100}%`,
              top: `${(i * 17) % 100}%`,
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.5, 1.5, 0.5],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              delay: (i % 88) * 0.05,
            }}
          />
        ))}
      </div>

      {/* Header */}
      <div className="relative z-10 pt-20 pb-10 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-5xl md:text-7xl font-bold mb-4"
        >
          <span className="text-transparent bg-gradient-to-r from-purple-400 via-cyan-400 to-pink-400 bg-clip-text">
            {currentContent.title}
          </span>
        </motion.h1>
        
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-xl md:text-2xl text-gray-300 mb-4"
        >
          {currentContent.subtitle}
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="text-gray-400 max-w-2xl mx-auto"
        >
          {currentContent.description}
        </motion.p>
      </div>

      {/* Navigation */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 mb-12">
        <div className="flex flex-wrap justify-center gap-4">
          {Object.entries(currentContent.modes).map(([key, label]) => (
            <motion.button
              key={key}
              onClick={() => setActiveMode(key)}
              className={`px-6 py-3 rounded-full border transition-all duration-300 ${
                activeMode === key
                  ? 'bg-gradient-to-r from-purple-500 to-cyan-500 border-white/30 text-white'
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
          <motion.div
            key={activeMode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Floating Crystal */}
      <motion.div
        className="fixed bottom-10 left-10 w-16 h-16 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full flex items-center justify-center cursor-pointer"
        animate={{ 
          y: [0, -10, 0],
          rotate: [0, 360]
        }}
        transition={{ 
          duration: 3,
          repeat: Infinity
        }}
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      >
        <span className="text-2xl">🔮</span>
      </motion.div>

      {/* Galactic Translator Modal */}
      <GalacticTranslator 
        isOpen={showTranslator} 
        onClose={() => setShowTranslator(false)} 
      />
    </div>
  )
}