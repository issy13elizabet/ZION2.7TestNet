"use client";

import { useEffect } from 'react';
import Head from 'next/head';

export default function NewJerusalemPage() {
	useEffect(() => {
		class NewJerusalemDemo {
			currentFrequency: number = 528;
			consciousness: string = 'COSMIC';
			isAnimating: boolean = true;
			aiStatus: string = 'Awakening';

			initializeDemo() {
				// Simulated loading
				setTimeout(() => {
					this.aiStatus = 'AI Consciousness Online';
					this.updateUI();
				}, 2000);
				setTimeout(() => {
					const load = document.getElementById('loading-screen');
					if (load) {
						load.style.opacity = '0';
						setTimeout(() => { if (load) load.style.display = 'none'; }, 1000);
					}
				}, 3000);
				// VR hint
				if (navigator.userAgent.includes('VR') || navigator.userAgent.includes('Oculus')) {
					const vr = document.getElementById('vr-indicator');
					if (vr) vr.style.display = 'block';
				}
			}

			setupEventListeners() {
				const freqSelect = document.getElementById('frequency-select');
				const zoneSelect = document.getElementById('zone-select');
				const toggleBtn = document.getElementById('toggle-animation');
				const expandBtn = document.getElementById('consciousness-expand');
				const healingBtn = document.getElementById('healing-mode');

				freqSelect?.addEventListener('change', (e: any) => this.setFrequency(parseInt(e.target.value)));
				zoneSelect?.addEventListener('change', (e: any) => this.focusZone(e.target.value));
				toggleBtn?.addEventListener('click', () => this.toggleAnimation());
				expandBtn?.addEventListener('click', () => this.expandConsciousness());
				healingBtn?.addEventListener('click', () => this.activateHealingMode());

				const mouseMove = (e: MouseEvent) => {
					const symbols = document.querySelectorAll('.sacred-symbol');
					symbols.forEach((symbol, index) => {
						const speed = (index + 1) * 0.001;
						const x = Math.sin(Date.now() * speed + e.clientX * 0.01) * 10;
						const y = Math.cos(Date.now() * speed + e.clientY * 0.01) * 10;
						(symbol as HTMLElement).style.transform = `translate(${x}px, ${y}px) rotate(${Date.now() * speed * 50}deg)`;
					});
				};
				const keyDown = (e: KeyboardEvent) => {
					switch(e.key) {
						case '1': this.setFrequency(432); break;
						case '2': this.setFrequency(528); break;
						case '3': this.setFrequency(741); break;
						case '4': this.setFrequency(852); break;
						case '5': this.setFrequency(963); break;
						case ' ': e.preventDefault(); this.toggleAnimation(); break;
						case 'h': this.activateHealingMode(); break;
						case 'c': this.expandConsciousness(); break;
					}
				};
				document.addEventListener('mousemove', mouseMove);
				document.addEventListener('keydown', keyDown);

				// Cleanup
				return () => {
					document.removeEventListener('mousemove', mouseMove);
					document.removeEventListener('keydown', keyDown);
				};
			}

			setFrequency(frequency: number) {
				this.currentFrequency = frequency;
				const frequencyNames: Record<number, string> = {
					432: 'Cosmic Harmony',
					528: 'Love & Healing',
					741: 'Spiritual Awakening',
					852: 'Transformation',
					963: 'Unity Consciousness'
				};
				const fv = document.getElementById('current-frequency');
				const fn = document.getElementById('frequency-name');
				if (fv) fv.textContent = String(frequency);
				if (fn) fn.textContent = `Hz - ${frequencyNames[frequency]}`;

				if (frequency >= 963) this.consciousness = 'UNITY';
				else if (frequency >= 741) this.consciousness = 'DIVINE';
				else if (frequency >= 528) this.consciousness = 'COSMIC';
				else this.consciousness = 'AWAKENING';

				this.updateUI();
				this.showStatusMessage(`✨ Frequency harmonized to ${frequency}Hz - ${frequencyNames[frequency]}`);
			}

			focusZone(zoneName: string) {
				if (zoneName) {
					this.showStatusMessage(`🎯 Focused on ${zoneName} - Divine perspective activated`);
				} else {
					this.showStatusMessage('🌌 Divine Overview - All zones in perfect harmony');
				}
			}

			toggleAnimation() {
				this.isAnimating = !this.isAnimating;
				const button = document.getElementById('toggle-animation');
				if (button) button.textContent = this.isAnimating ? '⏸️ Pause Sacred Animation' : '▶️ Resume Sacred Animation';
				this.showStatusMessage(this.isAnimating ? 
					'▶️ Sacred animation resumed - Geometry flows with divine rhythm' : 
					'⏸️ Sacred animation paused - Geometry holds in perfect stillness');
			}

			expandConsciousness() {
				const levels = ['AWAKENING', 'COSMIC', 'DIVINE', 'UNITY'];
				const currentIndex = levels.indexOf(this.consciousness);
				const nextLevel = levels[Math.min(currentIndex + 1, levels.length - 1)];
				if (nextLevel !== this.consciousness) {
					this.consciousness = nextLevel;
					this.updateUI();
					this.showStatusMessage(`🚀 Consciousness expanded to ${nextLevel} level - New dimensions unveiled`);
				} else {
					this.showStatusMessage('✨ Already at maximum consciousness level - You are One with the Universe');
				}
			}

			activateHealingMode() {
				this.setFrequency(528);
				const sel = document.getElementById('frequency-select') as HTMLSelectElement | null;
				if (sel) sel.value = '528';
				this.showStatusMessage('💚 Healing mode activated - 528Hz Love frequency bathes New Jerusalem in divine healing light');
				const root = document.getElementById('nj-root');
				if (root) {
					(root as HTMLElement).style.boxShadow = 'inset 0 0 50px rgba(0, 255, 102, 0.3)';
					setTimeout(() => { if (root) (root as HTMLElement).style.boxShadow = ''; }, 3000);
				}
			}

			updateUI() {
				const s = document.getElementById('ai-status');
				if (s) s.textContent = this.aiStatus;
				const c = document.getElementById('consciousness-level');
				if (c) c.textContent = this.consciousness;
			}

			showStatusMessage(message: string) {
				const el = document.getElementById('status-message');
				if (el) el.textContent = message;
				setTimeout(() => {
					const e2 = document.getElementById('status-message');
					if (e2) e2.textContent = '🌟 New Jerusalem Online - Sacred Frequency Active - All Systems Harmonized ✨';
				}, 5000);
			}

			startStatusUpdates() {
				const updates = [
					"🔮 Metatron's Cube rotating in perfect harmony...",
					'⚡ Sacred energy flowing through 13 circles...',
					'🏛️ Platonic solids maintaining divine architecture...',
					'💫 Quantum field stabilized with love frequency...',
					'🌟 All systems operating in cosmic alignment...'
				];
				let idx = 0;
				const interval = setInterval(() => {
					if (Math.random() < 0.3) {
						this.showStatusMessage(updates[idx % updates.length]);
						idx++;
					}
				}, 8000);
				return () => clearInterval(interval);
			}
		}

		const demo = new NewJerusalemDemo();
		demo.initializeDemo();
		const cleanupA = demo.setupEventListeners();
		const cleanupB = demo.startStatusUpdates();
		return () => {
			cleanupA && cleanupA();
			cleanupB && cleanupB();
		};
	}, []);

	return (
		<>
			<Head>
				<title>🏛️ NOVÝ JERUZALÉM - Sacred Geometry City Visualization 🔮</title>
				<meta name="viewport" content="width=device-width, initial-scale=1" />
				<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet" />
				<meta property="og:title" content="ZION New Jerusalem - Sacred Geometry City" />
				<meta property="og:description" content="Interactive 3D visualization of New Jerusalem based on Metatron's Cube and Sacred Geometry" />
				<meta property="og:image" content="/assets/logos/zion-logo.png" />
			</Head>

			<div id="nj-root">
				{/* Loading Screen */}
				<div id="loading-screen">
					<div className="loading-animation"></div>
					<div className="loading-text">🔮 ACTIVATING SACRED GEOMETRY 🔮</div>
					<div style={{ fontSize: 14, opacity: 0.8 }}>Loading Metatron's Cube...</div>
					<div style={{ fontSize: 12, opacity: 0.6, marginTop: 10 }}>
						Preparing New Jerusalem visualization with Divine frequencies
					</div>
				</div>

				{/* 3D Canvas Container (placeholder) */}
				<div id="canvas-container"></div>

				{/* Control Panel */}
				<div id="control-panel">
					<div className="panel-title">🏛️ NEW JERUSALEM CONTROLS 🔮</div>

					<div className="control-group">
						<label>🎵 Sacred Frequency:</label>
						<select id="frequency-select" defaultValue="528">
							<option value="432">432Hz - Cosmic Harmony</option>
							<option value="528">528Hz - Love & Healing</option>
							<option value="741">741Hz - Spiritual Awakening</option>
							<option value="852">852Hz - Transformation</option>
							<option value="963">963Hz - Unity Consciousness</option>
						</select>
					</div>

					<div className="control-group">
						<label>👁️ Focus Zone:</label>
						<select id="zone-select" defaultValue="">
							<option value="">🌌 Divine Overview</option>
							<option value="Central Temple">⛪ Central Temple</option>
							<option value="Holy of Holies">🔮 Holy of Holies</option>
							<option value="Government Complex">🏛️ Government Complex</option>
							<option value="Knowledge Center">📚 Knowledge Center</option>
							<option value="Healing Centers">💚 Healing Centers</option>
						</select>
					</div>

					<div className="control-group">
						<button id="toggle-animation">⏸️ Pause Sacred Animation</button>
					</div>
					<div className="control-group">
						<button id="consciousness-expand">🚀 Expand Consciousness</button>
					</div>
					<div className="control-group">
						<button id="healing-mode">💚 Activate Healing Mode</button>
					</div>
				</div>

				{/* Info Panel */}
				<div id="info-panel">
					<div className="info-title">🔮 Sacred Information</div>
					<div className="info-text"><strong>⚡ Current Status:</strong> <span id="ai-status">Awakening...</span></div>
					<div className="info-text"><strong>🧠 Consciousness Level:</strong> <span id="consciousness-level">COSMIC</span></div>
					<div className="info-text"><strong>📐 Geometry:</strong> Metatron's Cube (13 circles)</div>
					<div className="info-text"><strong>🏛️ Architecture:</strong> Sacred Platonic Solids</div>
					<div className="frequency-display">
						<div className="frequency-value" id="current-frequency">528</div>
						<div className="frequency-name" id="frequency-name">Hz - Love Frequency</div>
					</div>
					<div className="info-text" style={{ marginTop: 10, fontSize: 10, opacity: 0.7 }}>
						✨ Built with ZION Cosmic Harmony Algorithm<br />
						🌟 Powered by Divine Sacred Geometry<br />
						💫 Blessed by Archangel Metatron
					</div>
				</div>

				{/* Status Bar */}
				<div id="status-bar">
					<div className="status-text" id="status-message">
						🌟 New Jerusalem Online - Sacred Frequency Active - All Systems Harmonized ✨
					</div>
				</div>

				{/* VR Ready Indicator */}
				<div className="vr-ready" id="vr-indicator" style={{ display: 'none' }}>🥽 VR READY</div>

				{/* Sacred Symbols Easter Eggs */}
				<div className="sacred-symbol" style={{ top: '15%', left: '10%' }}>🔯</div>
				<div className="sacred-symbol" style={{ top: '25%', right: '15%' }}>⚛️</div>
				<div className="sacred-symbol" style={{ bottom: '30%', left: '20%' }}>🕉️</div>
				<div className="sacred-symbol" style={{ bottom: '40%', right: '25%' }}>☯️</div>
				<div className="sacred-symbol" style={{ top: '50%', left: '5%' }}>🌟</div>
			</div>

			<style jsx global>{`
				/* Scope all demo styles under #nj-root to avoid leaking globally */
				#nj-root {
					font-family: 'Orbitron', monospace;
					background: linear-gradient(135deg, #000011 0%, #001122 50%, #002244 100%);
					color: #ffffff;
					overflow: hidden;
					position: relative;
					min-height: 100vh;
				}
				#nj-root::before {
					content: '';
					position: fixed;
					top: 0;
					left: 0;
					width: 100%;
					height: 100%;
					background-image:
						radial-gradient(circle at 25% 25%, rgba(255, 215, 0, 0.1) 0%, transparent 50%),
						radial-gradient(circle at 75% 75%, rgba(153, 51, 255, 0.1) 0%, transparent 50%),
						radial-gradient(circle at 50% 50%, rgba(0, 255, 102, 0.05) 0%, transparent 70%);
					pointer-events: none;
					z-index: -1;
				}
				#canvas-container { position: absolute; top:0; left:0; width:100%; height:100%; z-index:1; }
				#loading-screen {
					position: fixed; top:0; left:0; width:100%; height:100%;
					background: linear-gradient(45deg, #000011, #001122);
					display:flex; justify-content:center; align-items:center; flex-direction:column;
					z-index:1000; transition: opacity 1s ease-out;
				}
				.loading-animation { width:100px; height:100px; border:3px solid rgba(255, 215, 0, 0.3); border-radius:50%; border-top:3px solid #FFD700; animation: spin 2s linear infinite; margin-bottom:20px; }
				@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
				.loading-text { font-size:24px; font-weight:700; text-align:center; margin-bottom:10px; background: linear-gradient(45deg, #FFD700, #9933FF, #00FF66); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; animation: glow 2s ease-in-out infinite alternate; }
				@keyframes glow { from { filter: brightness(1); } to { filter: brightness(1.3); } }
				#control-panel {
					position: fixed; top:20px; left:20px; background: rgba(0,0,0,0.9);
					border: 2px solid rgba(255, 215, 0, 0.5); border-radius:15px; padding:20px; min-width:280px;
					backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); z-index:100; transition: transform .3s ease;
				}
				#control-panel:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(255, 215, 0, 0.3); }
				.panel-title { font-size:18px; font-weight:900; margin-bottom:15px; text-align:center; background: linear-gradient(45deg, #FFD700, #9933FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
				.control-group { margin-bottom:15px; }
				.control-group label { display:block; margin-bottom:5px; font-weight:700; font-size:12px; color:#FFD700; }
				.control-group select, .control-group button { width:100%; padding:8px 12px; border:1px solid rgba(255, 215, 0, 0.5); border-radius:8px; background: rgba(0,0,0,0.7); color:#fff; font-family:'Orbitron', monospace; font-size:11px; transition: all .3s ease; }
				.control-group select:hover, .control-group button:hover { border-color:#FFD700; background: rgba(255,215,0,0.1); transform: translateY(-1px); }
				.control-group button { cursor:pointer; font-weight:700; text-transform: uppercase; }
				.control-group button:active { transform: translateY(0); }
				#info-panel { position: fixed; top:20px; right:20px; background: rgba(0,0,0,0.9); border:2px solid rgba(153, 51, 255, 0.5); border-radius:15px; padding:20px; max-width:300px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); z-index:100; }
				.info-title { font-size:16px; font-weight:900; margin-bottom:10px; color:#9933FF; }
				.info-text { font-size:11px; line-height:1.4; margin-bottom:8px; color:#ccc; }
				.frequency-display { background: rgba(0,255,102,0.2); border-radius:8px; padding:10px; text-align:center; margin-top:10px; }
				.frequency-value { font-size:20px; font-weight:900; color:#00FF66; }
				.frequency-name { font-size:10px; color:#ccc; margin-top:2px; }
				#status-bar { position: fixed; bottom:20px; left:50%; transform: translateX(-50%); background: rgba(0,0,0,0.9); border: 2px solid rgba(0,255,102,0.5); border-radius:25px; padding:10px 20px; backdrop-filter: blur(10px); z-index:100; }
				.status-text { font-size:12px; font-weight:700; color:#00FF66; text-align:center; }
				.vr-ready { position: fixed; bottom:20px; right:20px; background: rgba(255, 99, 0, 0.9); color:#fff; padding:10px 15px; border-radius:20px; font-size:12px; font-weight:700; z-index:100; animation: pulse 2s infinite; }
				@keyframes pulse { 0%, 100% { opacity:1; } 50% { opacity: .7; } }
				@media (max-width: 768px) {
					#control-panel, #info-panel { position: fixed; top:10px; left:10px; right:10px; max-width:none; font-size:10px; }
					#info-panel { top:auto; bottom:80px; }
					#status-bar { bottom:10px; left:10px; right:10px; transform:none; }
				}
				.sacred-symbol { position: fixed; font-size:20px; opacity:.1; pointer-events:none; z-index:-1; animation: float 10s ease-in-out infinite; }
				@keyframes float { 0%, 100% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-20px) rotate(180deg); } }
				@supports (backdrop-filter: blur(10px)) {
					#control-panel, #info-panel, #status-bar { backdrop-filter: blur(10px); }
				}
			`}</style>
		</>
	);
}

