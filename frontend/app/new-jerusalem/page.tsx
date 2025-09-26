'use client'

import { Suspense, useEffect, useState, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Sphere, Box, Cylinder } from '@react-three/drei'
import * as THREE from 'three'
import { motion } from 'framer-motion'

// 🏛️ Sacred Colors (Chakra aligned)
const SACRED_COLORS = {
  VIOLET: '#9933FF',    // Crown Chakra - Central Temple
  INDIGO: '#4B0082',    // Third Eye - Wisdom Centers
  BLUE: '#0066FF',      // Throat - Communication
  GREEN: '#00FF66',     // Heart - Healing Centers
  YELLOW: '#FFFF00',    // Solar Plexus - Education
  ORANGE: '#FF6600',    // Sacral - Residential
  RED: '#FF0033',       // Root - Foundation
  WHITE: '#FFFFFF',     // Unity - Pure Light
  GOLD: '#FFD700'       // Divine - Sacred Elements
} as const

// Sacred Frequencies
const SACRED_FREQUENCIES = {
  SOLFEGGIO_396: 396,   // Liberation from Fear
  SOLFEGGIO_417: 417,   // Facilitating Change
  SOLFEGGIO_528: 528,   // Love & DNA Repair
  SOLFEGGIO_639: 639,   // Connecting & Relationships
  SOLFEGGIO_741: 741,   // Awakening Intuition
  SOLFEGGIO_852: 852,   // Returning to Spiritual Order
  SOLFEGGIO_963: 963    // Divine Consciousness
} as const

// 🔮 Metatron's Cube Sacred Geometry
function MetatronCube({ frequency }: { frequency: number }) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const [time, setTime] = useState(0)

  useFrame((state, delta) => {
    setTime(time + delta)
    if (meshRef.current) {
      meshRef.current.rotation.x = time * 0.3
      meshRef.current.rotation.y = time * 0.5
      meshRef.current.rotation.z = time * 0.1
      
      // Frequency-based pulsing
      const scale = 1 + Math.sin(time * (frequency / 100)) * 0.1
      meshRef.current.scale.setScalar(scale)
    }
  })

  return (
    <group ref={meshRef}>
      {/* Central sphere - Unity */}
      <Sphere args={[2, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color={SACRED_COLORS.GOLD} 
          emissive={SACRED_COLORS.YELLOW}
          emissiveIntensity={0.2}
          transparent
          opacity={0.8}
        />
      </Sphere>
      
      {/* Sacred vertices - 13 spheres of Metatron's Cube */}
      {[
        [0, 5, 0], [0, -5, 0],           // Top & Bottom
        [4.33, 2.5, 0], [-4.33, 2.5, 0], // Upper ring
        [4.33, -2.5, 0], [-4.33, -2.5, 0], // Lower ring
        [2.17, 1.25, 3.75], [-2.17, 1.25, 3.75], // Front upper
        [2.17, -1.25, 3.75], [-2.17, -1.25, 3.75], // Front lower
        [2.17, 1.25, -3.75], [-2.17, 1.25, -3.75], // Back upper
        [2.17, -1.25, -3.75], [-2.17, -1.25, -3.75] // Back lower
      ].map((pos, i) => (
        <Sphere key={i} args={[0.5, 16, 16]} position={pos as [number, number, number]}>
          <meshStandardMaterial 
            color={Object.values(SACRED_COLORS)[i % Object.keys(SACRED_COLORS).length]} 
            emissive={Object.values(SACRED_COLORS)[i % Object.keys(SACRED_COLORS).length]}
            emissiveIntensity={0.3}
          />
        </Sphere>
      ))}
    </group>
  )
}

// 🏛️ Sacred Temple Zone
function TempleZone({ 
  position, 
  color, 
  radius, 
  height, 
  name 
}: {
  position: [number, number, number]
  color: string
  radius: number
  height: number
  name: string
}) {
  const meshRef = useRef<THREE.Mesh>(null!)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.2
    }
  })

  return (
    <group position={position}>
      {/* Zone base ring */}
      <Cylinder 
        ref={meshRef}
        args={[radius, radius, height, 32, 1, true]}
      >
        <meshStandardMaterial 
          color={color}
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
        />
      </Cylinder>
      
      {/* Zone label */}
      <Text
        position={[0, height + 2, 0]}
        fontSize={1}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        {name}
      </Text>
      
      {/* Sacred buildings in zone */}
      {Array.from({ length: 8 }, (_, i) => {
        const angle = (i / 8) * Math.PI * 2
        const buildingX = Math.cos(angle) * (radius - 1)
        const buildingZ = Math.sin(angle) * (radius - 1)
        const buildingHeight = 2 + Math.random() * 3
        
        return (
          <Box 
            key={i}
            position={[buildingX, buildingHeight / 2, buildingZ]}
            args={[0.8, buildingHeight, 0.8]}
          >
            <meshStandardMaterial 
              color={color}
              emissive={color}
              emissiveIntensity={0.2}
            />
          </Box>
        )
      })}
    </group>
  )
}

// 🌟 New Jerusalem City
function NewJerusalemCity({ frequency }: { frequency: number }) {
  return (
    <group>
      {/* Central Metatron's Cube */}
      <MetatronCube frequency={frequency} />
      
      {/* Sacred Temple Zones */}
      <TempleZone
        position={[0, -1, 0]}
        color={SACRED_COLORS.VIOLET}
        radius={8}
        height={1}
        name="Holy of Holies"
      />
      
      <TempleZone
        position={[0, -2, 0]}
        color={SACRED_COLORS.BLUE}
        radius={12}
        height={1}
        name="Healing Sanctuary"
      />
      
      <TempleZone
        position={[0, -3, 0]}
        color={SACRED_COLORS.GREEN}
        radius={16}
        height={1}
        name="Wisdom Council"
      />
      
      <TempleZone
        position={[0, -4, 0]}
        color={SACRED_COLORS.YELLOW}
        radius={20}
        height={1}
        name="Learning Pavilion"
      />
      
      <TempleZone
        position={[0, -5, 0]}
        color={SACRED_COLORS.ORANGE}
        radius={24}
        height={1}
        name="Unity Circle"
      />
      
      <TempleZone
        position={[0, -6, 0]}
        color={SACRED_COLORS.RED}
        radius={28}
        height={1}
        name="Protection Field"
      />
    </group>
  )
}

// 🎵 Audio Controller for Sacred Frequencies
function AudioController({ frequency, isPlaying }: { frequency: number, isPlaying: boolean }) {
  useEffect(() => {
    if (!isPlaying) return

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)

    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime)
    oscillator.type = 'sine'
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime)

    oscillator.start()

    return () => {
      oscillator.stop()
      audioContext.close()
    }
  }, [frequency, isPlaying])

  return null
}

// 🔮 Main New Jerusalem Museum Page
export default function NewJerusalemMuseum() {
  const [currentFrequency, setCurrentFrequency] = useState(SACRED_FREQUENCIES.SOLFEGGIO_528)
  const [isAudioPlaying, setIsAudioPlaying] = useState(false)
  const [consciousnessLevel, setConsciousnessLevel] = useState(1)
  const [selectedZone, setSelectedZone] = useState('holy_of_holies')

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-900 via-purple-900 to-black text-white">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 text-center border-b border-gold-500/30"
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-gold-400 to-purple-400 bg-clip-text text-transparent">
          🏛️ NEW JERUSALEM MUSEUM 🔮
        </h1>
        <p className="mt-2 text-lg text-gray-300">
          Interactive Sacred Geometry City • VR Ready • Metatron&apos;s Cube Architecture
        </p>
      </motion.div>

      <div className="flex flex-col lg:flex-row h-screen">
        {/* Control Panel */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:w-80 p-6 bg-black/50 backdrop-blur-md border-r border-gold-500/30"
        >
          <div className="space-y-6">
            {/* Sacred Frequency Control */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gold-400">🎵 Sacred Frequencies</h3>
              <select 
                value={currentFrequency}
                onChange={(e) => setCurrentFrequency(Number(e.target.value))}
                className="w-full p-2 bg-gray-800 border border-gold-500/30 rounded-lg text-white"
              >
                {Object.entries(SACRED_FREQUENCIES).map(([name, freq]) => (
                  <option key={name} value={freq}>
                    {freq} Hz - {name.replace('SOLFEGGIO_', '').replace('_', ' ')}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setIsAudioPlaying(!isAudioPlaying)}
                className={`w-full p-2 rounded-lg font-semibold ${
                  isAudioPlaying 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                {isAudioPlaying ? '🔇 Stop Frequency' : '🎵 Play Frequency'}
              </button>
            </div>

            {/* Consciousness Level */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gold-400">🧘 Consciousness Level</h3>
              <input
                type="range"
                min="1"
                max="7"
                value={consciousnessLevel}
                onChange={(e) => setConsciousnessLevel(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm">
                Level {consciousnessLevel}: {
                  ['', 'Survival', 'Emotional', 'Personal Power', 'Love', 'Truth', 'Intuition', 'Divine Unity'][consciousnessLevel]
                }
              </div>
            </div>

            {/* Zone Selection */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gold-400">🏛️ Temple Zones</h3>
              <select 
                value={selectedZone}
                onChange={(e) => setSelectedZone(e.target.value)}
                className="w-full p-2 bg-gray-800 border border-gold-500/30 rounded-lg text-white"
              >
                <option value="holy_of_holies">Holy of Holies</option>
                <option value="healing_sanctuary">Healing Sanctuary</option>
                <option value="wisdom_council">Wisdom Council</option>
                <option value="learning_pavilion">Learning Pavilion</option>
                <option value="unity_circle">Unity Circle</option>
                <option value="protection_field">Protection Field</option>
              </select>
            </div>

            {/* Status Display */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-gold-400">📊 Temple Status</h3>
              <div className="text-sm space-y-1">
                <div>🎵 Frequency: {currentFrequency} Hz</div>
                <div>🧘 Consciousness: Level {consciousnessLevel}</div>
                <div>🔮 Zone: {selectedZone.replace('_', ' ').toUpperCase()}</div>
                <div>✨ AI Status: Metatron Online</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* 3D Canvas */}
        <div className="flex-1 relative">
          <Canvas
            camera={{ position: [50, 30, 50], fov: 75 }}
            gl={{ antialias: true }}
          >
            <Suspense fallback={null}>
              <ambientLight intensity={0.4} />
              <pointLight position={[10, 10, 10]} intensity={1} />
              <pointLight position={[-10, -10, -10]} intensity={0.5} color="#9933FF" />
              
              <Stars 
                radius={300} 
                depth={50} 
                count={5000} 
                factor={4} 
                saturation={0} 
                fade 
              />
              
              <NewJerusalemCity frequency={currentFrequency} />
              
              <OrbitControls 
                enableDamping 
                dampingFactor={0.05}
                maxDistance={200}
                minDistance={5}
              />
            </Suspense>
          </Canvas>

          {/* Audio Controller */}
          <AudioController frequency={currentFrequency} isPlaying={isAudioPlaying} />

          {/* Floating Info */}
          <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-md p-4 rounded-lg border border-gold-500/30">
            <div className="text-sm space-y-1">
              <div className="font-semibold text-gold-400">🔮 Sacred Geometry Active</div>
              <div>Frequency: {currentFrequency} Hz</div>
              <div>Temple Energy: 77%</div>
              <div>Visitors: {Math.floor(Math.random() * 144)} souls</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}