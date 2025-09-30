'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { galacticAlphabets } from './galactic-data'

interface GalacticTranslatorProps {
  isOpen: boolean
  onClose: () => void
}

export default function GalacticTranslator({ isOpen, onClose }: GalacticTranslatorProps) {
  const [inputText, setInputText] = useState('')
  const [selectedAlphabet, setSelectedAlphabet] = useState('pleiadian')
  const [translatedText, setTranslatedText] = useState('')

  const translateToGalactic = (text: string, alphabet: string) => {
    const symbols = galacticAlphabets[alphabet as keyof typeof galacticAlphabets].symbols
    
    return text
      .toUpperCase()
      .split('')
      .map(char => {
        if (char === ' ') return '   '
        return symbols[char as keyof typeof symbols] || char
      })
      .join(' ')
  }

  const handleTranslate = () => {
    const translated = translateToGalactic(inputText, selectedAlphabet)
    setTranslatedText(translated)
  }

  if (!isOpen) return null

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
    >
      <motion.div
        initial={{ y: 50 }}
        animate={{ y: 0 }}
        className="bg-gradient-to-br from-purple-900/90 to-blue-900/90 rounded-2xl p-8 max-w-2xl w-full border border-purple-500/30"
      >
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">
            🔮 Galactic Text Translator
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ✕
          </button>
        </div>

        {/* Alphabet Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Select Galactic Alphabet:
          </label>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(galacticAlphabets).map(([key, alphabet]) => (
              <button
                key={key}
                onClick={() => setSelectedAlphabet(key)}
                className={`p-3 rounded-lg text-left transition-all ${
                  selectedAlphabet === key
                    ? 'bg-purple-600 text-white border-purple-400'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border-gray-600'
                } border`}
              >
                <div className="font-semibold">⭐ {alphabet.name}</div>
                <div className="text-sm opacity-75">
                  {key.charAt(0).toUpperCase() + key.slice(1)} origin
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Earth Text (English):
          </label>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter your message in English..."
            className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 h-24 resize-none"
            maxLength={100}
          />
          <div className="text-xs text-gray-400 mt-1">
            {inputText.length}/100 characters
          </div>
        </div>

        {/* Translate Button */}
        <button
          onClick={handleTranslate}
          disabled={!inputText.trim()}
          className="w-full py-3 mb-4 bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-cyan-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          🚀 Translate to Galactic
        </button>

        {/* Output */}
        {translatedText && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4"
          >
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Galactic Translation ({galacticAlphabets[selectedAlphabet as keyof typeof galacticAlphabets].name}):
            </label>
            <div className="p-4 bg-black/30 border border-purple-500/30 rounded-lg">
              <div className="text-2xl text-cyan-300 font-mono leading-relaxed text-center">
                {translatedText}
              </div>
            </div>
            
            <button
              onClick={() => navigator.clipboard?.writeText(translatedText)}
              className="mt-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-all"
            >
              📋 Copy Galactic Text
            </button>
          </motion.div>
        )}

        {/* Sample Alphabet */}
        <div className="border-t border-gray-600 pt-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">
            {galacticAlphabets[selectedAlphabet as keyof typeof galacticAlphabets].name} Alphabet:
          </h3>
          <div className="grid grid-cols-7 gap-2 text-sm">
            {Object.entries(galacticAlphabets[selectedAlphabet as keyof typeof galacticAlphabets].symbols).map(([letter, symbol]) => (
              <div key={letter} className="text-center p-2 bg-gray-800 rounded">
                <div className="text-gray-400 text-xs">{letter}</div>
                <div className="text-cyan-300 text-lg">{symbol}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}