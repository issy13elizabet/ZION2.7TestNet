'use client';

import { Suspense, useRef, useMemo } from 'react';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import Link from 'next/link';
import dynamic from 'next/dynamic';

// Import useFrame for animations
let useFrame: any = null;
if (typeof window !== 'undefined') {
  import('@react-three/fiber').then(mod => {
    useFrame = mod.useFrame;
  });
}

// Dynamically import Canvas to avoid SSR issues
const Canvas = dynamic(() => import('@react-three/fiber').then(mod => ({ default: mod.Canvas })), { ssr: false });
const OrbitControls = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.OrbitControls })), { ssr: false });
const Text = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Text })), { ssr: false });
const Float = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Float })), { ssr: false });
const Sphere = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Sphere })), { ssr: false });
const Box = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Box })), { ssr: false });
const Cylinder = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Cylinder })), { ssr: false });
const Environment = dynamic(() => import('@react-three/drei').then(mod => ({ default: mod.Environment })), { ssr: false });

// Sacred Geometry Components
function SacredPillar({ position, height = 8, color = "#FFD700" }: { position: [number, number, number], height?: number, color?: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state: any) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <Float speed={1.4} rotationIntensity={0.2} floatIntensity={0.3}>
      <group position={position}>
        {/* Main Pillar */}
        <Cylinder ref={meshRef} args={[0.5, 0.6, height, 8]} position={[0, height/2, 0]}>
          <meshStandardMaterial color={color} metalness={0.7} roughness={0.3} />
        </Cylinder>
        
        {/* Sacred Symbols on Pillar */}
        {[...Array(4)].map((_, i) => (
          <Text
            key={i}
            position={[0, 2 + i * 2, 0.6]}
            rotation={[0, (i * Math.PI) / 2, 0]}
            fontSize={0.5}
            color="#FF6B35"
            anchorX="center"
            anchorY="middle"
          >
            🕉️
          </Text>
        ))}
        
        {/* Crystal Top */}
        <Box args={[1, 1, 1]} position={[0, height + 1, 0]} rotation={[0, Math.PI/4, 0]}>
          <meshPhysicalMaterial color="#E6E6FA" transmission={0.9} opacity={0.8} roughness={0.1} />
        </Box>
      </group>
    </Float>
  );
}

function FloatingOm({ position }: { position: [number, number, number] }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state: any) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.8;
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.5;
    }
  });

  return (
    <Text
      ref={meshRef}
      position={position}
      fontSize={2}
      color="#FFD700"
      anchorX="center"
      anchorY="middle"
    >
      🕉️
    </Text>
  );
}

function CrystalFormation({ position, count = 5 }: { position: [number, number, number], count?: number }) {
  const crystals = useMemo(() => {
    return [...Array(count)].map((_, i) => ({
      position: [
        position[0] + (Math.random() - 0.5) * 4,
        position[1] + Math.random() * 2,
        position[2] + (Math.random() - 0.5) * 4,
      ] as [number, number, number],
      rotation: [Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI] as [number, number, number],
      scale: 0.5 + Math.random() * 1.5,
      color: ['#FF6B35', '#FFD700', '#E6E6FA', '#87CEEB'][Math.floor(Math.random() * 4)]
    }));
  }, [position, count]);

  return (
    <group>
      {crystals.map((crystal, i) => (
        <Float key={i} speed={1 + i * 0.2} rotationIntensity={0.1} floatIntensity={0.2}>
          <Box 
            args={[0.5, 2, 0.5]} 
            position={crystal.position} 
            rotation={crystal.rotation}
            scale={crystal.scale}
          >
            <meshPhysicalMaterial 
              color={crystal.color} 
              transmission={0.7} 
              opacity={0.9} 
              roughness={0.1}
              metalness={0.3}
            />
          </Box>
        </Float>
      ))}
    </group>
  );
}

function TempleFloor() {
  return (
    <group>
      {/* Main Floor */}
      <Cylinder args={[25, 25, 0.5, 32]} position={[0, -0.25, 0]}>
        <meshStandardMaterial color="#8B4513" metalness={0.2} roughness={0.8} />
      </Cylinder>
      
      {/* Sacred Geometry Pattern */}
      {[...Array(3)].map((_, ring) => (
        <group key={ring}>
          {[...Array(12)].map((_, i) => {
            const angle = (i / 12) * Math.PI * 2;
            const radius = 5 + ring * 5;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            
            return (
              <Cylinder 
                key={`${ring}-${i}`}
                args={[0.3, 0.3, 0.1, 8]} 
                position={[x, 0.1, z]}
              >
                <meshStandardMaterial color="#FFD700" metalness={0.8} roughness={0.2} />
              </Cylinder>
            );
          })}
        </group>
      ))}
    </group>
  );
}

function TempleDome() {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state: any) => {
    if (meshRef.current) {
      const mat = (meshRef.current as THREE.Mesh).material as THREE.MeshStandardMaterial;
      if (mat) {
        // guard for standard material
        (mat as any).emissiveIntensity = 0.2 + Math.sin(state.clock.elapsedTime) * 0.1;
      }
    }
  });

  return (
    <Sphere ref={meshRef} args={[30, 32, 32]} position={[0, 20, 0]} scale={[1, 0.5, 1]}>
      <meshStandardMaterial 
        color="#4169E1" 
        transparent 
        opacity={0.3} 
        side={THREE.DoubleSide}
        emissive="#4169E1"
        emissiveIntensity={0.2}
      />
    </Sphere>
  );
}

function ParticleField() {
  const points = useMemo(() => {
    const temp = [];
    for (let i = 0; i < 1000; i++) {
      temp.push(
        (Math.random() - 0.5) * 100,
        Math.random() * 50,
        (Math.random() - 0.5) * 100
      );
    }
    return new Float32Array(temp);
  }, []);

  const pointsRef = useRef<THREE.Points>(null);

  useFrame((state: any) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.05;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[points, 3]}
          count={points.length / 3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} color="#FFD700" transparent opacity={0.6} />
    </points>
  );
}

export default function VirtualTemple3D() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      
      {/* Navigation */}
      <motion.nav
        className="relative z-20 p-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/ekam" className="text-amber-300 hover:text-white transition-colors text-lg">
          ← Back to EKAM Temple
        </Link>
      </motion.nav>

      {/* Header */}
      <motion.div 
        className="relative z-20 text-center px-6 mb-4"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h1 className="text-6xl font-bold bg-gradient-to-r from-amber-400 via-orange-300 to-red-400 bg-clip-text text-transparent mb-4">
          🏛️ Virtual EKAM Temple
        </h1>
        <p className="text-xl text-amber-300 mb-2">
          Main Temple Hall • Sacred 3D Experience ॐ
        </p>
        <p className="text-amber-200/80">
          Use mouse to orbit • Scroll to zoom • Experience the sacred geometry
        </p>
      </motion.div>

      {/* 3D Canvas */}
      <div className="h-[70vh] w-full">
        <Suspense fallback={
          <div className="h-full w-full flex items-center justify-center bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 rounded-lg">
            <div className="text-center">
              <motion.div
                className="text-6xl mb-4"
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                🕉️
              </motion.div>
              <p className="text-amber-300 text-xl">Loading Sacred Temple...</p>
              <p className="text-amber-200 text-sm mt-2">Awakening 3D Consciousness ✨</p>
            </div>
          </div>
        }>
          <Canvas camera={{ position: [0, 10, 20], fov: 60 }}>
            <color attach="background" args={['#0B0B2F']} />
          
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <pointLight position={[0, 15, 0]} intensity={1} color="#FFD700" />
          <spotLight 
            position={[10, 20, 10]} 
            angle={0.3} 
            penumbra={0.5} 
            intensity={0.5}
            color="#FF6B35"
            castShadow
          />

          {/* Environment */}
          <Environment preset="night" />
          
          {/* Temple Components */}
          <TempleFloor />
          <TempleDome />
          <ParticleField />
          
          {/* Sacred Pillars in Circle */}
          {[...Array(8)].map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            const radius = 12;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            
            return (
              <SacredPillar 
                key={i} 
                position={[x, 0, z]} 
                height={10}
                color={['#FFD700', '#FF6B35', '#E6E6FA', '#87CEEB'][i % 4]}
              />
            );
          })}
          
          {/* Central Altar */}
          <group position={[0, 0, 0]}>
            <Float speed={0.5} rotationIntensity={0.1} floatIntensity={0.1}>
              <Cylinder args={[3, 3, 1, 16]} position={[0, 0.5, 0]}>
                <meshStandardMaterial color="#8B4513" metalness={0.3} roughness={0.7} />
              </Cylinder>
              
              {/* Sacred OM in center */}
              <FloatingOm position={[0, 2, 0]} />
            </Float>
          </group>
          
          {/* Crystal Formations */}
          <CrystalFormation position={[-15, 1, -15]} count={7} />
          <CrystalFormation position={[15, 1, 15]} count={7} />
          <CrystalFormation position={[-15, 1, 15]} count={7} />
          <CrystalFormation position={[15, 1, -15]} count={7} />
          
          {/* Floating Sacred Symbols */}
          {[...Array(12)].map((_, i) => {
            const angle = (i / 12) * Math.PI * 2;
            const radius = 8;
            const x = Math.cos(angle) * radius;
            const z = Math.sin(angle) * radius;
            
            return (
              <FloatingOm 
                key={`symbol-${i}`}
                position={[x, 4 + Math.sin(i) * 2, z]} 
              />
            );
          })}

          {/* Controls */}
          <OrbitControls 
            enablePan={true} 
            enableZoom={true} 
            enableRotate={true}
            maxDistance={50}
            minDistance={5}
            maxPolarAngle={Math.PI / 2.2}
          />
        </Canvas>
        </Suspense>
      </div>

      {/* Interactive Controls */}
      <motion.div 
        className="relative z-20 text-center px-6 py-8"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div className="max-w-4xl mx-auto">
          <h3 className="text-2xl font-semibold text-amber-300 mb-6">🔮 Sacred Temple Features</h3>
          
          <div className="grid md:grid-cols-3 gap-6">
            <motion.div 
              className="bg-black/30 backdrop-blur-sm p-6 rounded-xl border border-amber-500/30"
              whileHover={{ scale: 1.02 }}
            >
              <div className="text-3xl mb-3">🏛️</div>
              <h4 className="text-amber-300 font-semibold mb-2">Sacred Architecture</h4>
              <p className="text-amber-100 text-sm">
                8 pillars representing cosmic directions with rotating sacred geometry
              </p>
            </motion.div>
            
            <motion.div 
              className="bg-black/30 backdrop-blur-sm p-6 rounded-xl border border-orange-500/30"
              whileHover={{ scale: 1.02 }}
            >
              <div className="text-3xl mb-3">🕉️</div>
              <h4 className="text-orange-300 font-semibold mb-2">Floating Mantras</h4>
              <p className="text-orange-100 text-sm">
                Sacred OM symbols floating in divine geometric patterns
              </p>
            </motion.div>
            
            <motion.div 
              className="bg-black/30 backdrop-blur-sm p-6 rounded-xl border border-red-500/30"
              whileHover={{ scale: 1.02 }}
            >
              <div className="text-3xl mb-3">💎</div>
              <h4 className="text-red-300 font-semibold mb-2">Crystal Energy</h4>
              <p className="text-red-100 text-sm">
                Healing crystals emanating light and consciousness frequencies
              </p>
            </motion.div>
          </div>

          <div className="mt-8 space-y-4">
            <p className="text-amber-200">
              🌟 <strong>Pro tip:</strong> Click and drag to orbit around the temple
            </p>
            <p className="text-orange-200">
              🔄 <strong>Scroll wheel:</strong> Zoom in to see sacred details up close
            </p>
            <p className="text-red-200">
              ✨ <strong>Meditation mode:</strong> Find the central altar and focus on the floating OM
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}