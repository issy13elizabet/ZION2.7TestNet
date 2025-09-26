"use client";

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function NewJerusalemMuseum() {
	const [selectedZone, setSelectedZone] = useState<string>('center');
	const [isPlaying, setIsPlaying] = useState(false);

	const zones = [
		{ id: 'center', name: 'Sacred Center', color: '#FFD700' },
		{ id: 'north', name: 'Divine Wisdom', color: '#9933FF' },
		{ id: 'south', name: 'Heart Resonance', color: '#00FF00' },
		{ id: 'east', name: 'New Beginnings', color: '#FFFF00' },
		{ id: 'west', name: 'Inner Peace', color: '#0066FF' }
	];

	return (
		<div className="min-h-screen bg-gradient-to-b from-black via-indigo-900 to-purple-900 text-white">
			<motion.div 
				initial={{ opacity: 0, y: -50 }}
				animate={{ opacity: 1, y: 0 }}
				className="text-center py-8"
			>
				<h1 className="text-5xl font-bold bg-gradient-to-r from-yellow-400 via-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">
					🏛️ NEW JERUSALEM MUSEUM 🔮
				</h1>
				<p className="text-xl text-gray-300">
					Interactive Sacred Geometry • Divine Consciousness Exploration
				</p>
			</motion.div>

			<div className="container mx-auto px-4 pb-8">
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
					<motion.div 
						initial={{ opacity: 0, scale: 0.8 }}
						animate={{ opacity: 1, scale: 1 }}
						className="bg-black/30 rounded-2xl p-8 backdrop-blur-sm border border-purple-500/30"
					>
						<h2 className="text-2xl font-bold mb-6 text-center text-yellow-400">
							Metatron's Cube - Sacred City Layout
						</h2>
            
						<div className="relative w-full h-96 flex items-center justify-center">
              
							<motion.div
								className="absolute w-64 h-64 border-2 border-yellow-400 rounded-full"
								animate={{ rotate: 360 }}
								transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
							/>
              
							<motion.div
								className="absolute w-32 h-32 rounded-full"
								animate={{ rotate: -360 }}
								transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
								style={{
									background: 'conic-gradient(from 0deg, #FF0000, #FF6600, #FFFF00, #00FF00, #0066FF, #4B0082, #9933FF, #FF0000)',
								}}
							/>

							{zones.map((zone, index) => {
								const angle = (index * 72) * (Math.PI / 180);
								const radius = 120;
								const x = Math.cos(angle) * radius;
								const y = Math.sin(angle) * radius;
                
								return (
									<motion.div
										key={zone.id}
										className="absolute w-8 h-8 rounded-full cursor-pointer"
										style={{ 
											left: `calc(50% + ${x}px)`,
											top: `calc(50% + ${y}px)`,
											backgroundColor: zone.color,
											transform: 'translate(-50%, -50%)'
										}}
										whileHover={{ scale: 1.5 }}
										onClick={() => {
											setSelectedZone(zone.id);
											setIsPlaying(true);
											setTimeout(() => setIsPlaying(false), 3000);
										}}
									/>
								);
							})}
						</div>
            
						<AnimatePresence>
							{selectedZone && (
								<motion.div
									initial={{ opacity: 0, y: 20 }}
									animate={{ opacity: 1, y: 0 }}
									exit={{ opacity: 0, y: -20 }}
									className="mt-6 p-4 bg-black/50 rounded-lg"
								>
									<h3 className="text-lg font-semibold text-yellow-400">
										Active Zone: {zones.find(z => z.id === selectedZone)?.name}
									</h3>
								</motion.div>
							)}
						</AnimatePresence>
					</motion.div>

					<div className="space-y-6">
						<motion.div 
							initial={{ opacity: 0, x: 50 }}
							animate={{ opacity: 1, x: 0 }}
							className="bg-black/30 rounded-2xl p-6 backdrop-blur-sm border border-purple-500/30"
						>
							<h3 className="text-xl font-bold mb-4 text-yellow-400">Sacred Zones</h3>
							<div className="space-y-3">
								{zones.map((zone) => (
									<motion.button
										key={zone.id}
										className="w-full p-3 rounded-lg text-left bg-purple-800/30 border border-purple-500/50 hover:bg-purple-700/40"
										onClick={() => setSelectedZone(zone.id)}
										whileHover={{ scale: 1.02 }}
									>
										<div className="flex items-center space-x-3">
											<div 
												className="w-4 h-4 rounded-full"
												style={{ backgroundColor: zone.color }}
											/>
											<span>{zone.name}</span>
										</div>
									</motion.button>
								))}
							</div>
						</motion.div>

						<motion.div 
							className="bg-black/30 rounded-2xl p-6 backdrop-blur-sm border border-purple-500/30"
						>
							<h3 className="text-xl font-bold mb-4 text-yellow-400">Status</h3>
							<div className="text-center">
								<span className={`text-2xl ${isPlaying ? 'text-green-400' : 'text-gray-400'}`}>
									{isPlaying ? '🎵 Active' : '⏸️ Silent'}
								</span>
							</div>
						</motion.div>
					</div>
				</div>

				<motion.div 
					initial={{ opacity: 0, y: 50 }}
					animate={{ opacity: 1, y: 0 }}
					transition={{ delay: 0.8 }}
					className="mt-8 text-center text-gray-400 text-sm"
				>
					<p>🔮 Interactive Sacred Geometry Visualization • Based on Metatron's Cube 🔮</p>
				</motion.div>
			</div>
		</div>
	);
}

