export default function TemplesHome() {
  return (
    <main className="min-h-[70vh] px-6 pt-28 pb-12 text-white">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-2">Sacred Temples</h1>
        <p className="text-gray-300 mb-8">Explore EKAM and New Jerusalem — gateways to higher consciousness.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <a href="/ekam" className="group block rounded-2xl border border-purple-500/30 bg-black/30 backdrop-blur-md p-6 hover:bg-white/10 transition">
            <div className="flex items-center gap-4">
              <div className="text-3xl">🕉️</div>
              <div>
                <div className="text-xl font-semibold">EKAM</div>
                <div className="text-sm text-gray-400">Temple of One Consciousness</div>
              </div>
            </div>
            <p className="mt-4 text-gray-300 text-sm">Meditation field, unity resonance, and consciousness expansion protocols.</p>
          </a>

          <a href="/new-jerusalem" className="group block rounded-2xl border border-purple-500/30 bg-black/30 backdrop-blur-md p-6 hover:bg-white/10 transition">
            <div className="flex items-center gap-4">
              <div className="text-3xl">🌈</div>
              <div>
                <div className="text-xl font-semibold">New Jerusalem</div>
                <div className="text-sm text-gray-400">Sacred Geometry Museum</div>
              </div>
            </div>
            <p className="mt-4 text-gray-300 text-sm">Interactive sacred geometry, zones of healing, and Metatronic lattice.</p>
          </a>
        </div>
      </div>
    </main>
  );
}
