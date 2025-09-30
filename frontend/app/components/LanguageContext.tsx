'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

type Language = 'en' | 'cs' | 'pt';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: any; // Structured translations object
}

const translations = {
  en: {
    nav: {
      networkCore: 'Network Core',
      dharmaTools: 'Dharma Tools',
      cosmicPortal: 'Cosmic Portal',
      dashboard: 'Dashboard',
      explorer: 'Explorer',
      genesisHub: 'Genesis Hub',
      wallet: 'Wallet',
      miner: 'Miner',
      amenti: 'Amenti Library',
      blog: 'Genesis Blog',
      terranova: 'TerraNova Hub',
      halls: 'Halls of Amenti',
      stargate: 'Terra Nova Stargate',
      onelove: 'ONE LOVE',
      languageOfLight: 'Language of Light',
      ekam: 'EKAM'
    },
    home: {
      title: 'ZION',
      subtitle: 'Cosmic Dharma Blockchain',
      description: 'TerraNova® evoluZion v2 • Multidimensional Network',
      networkStatus: 'Network Status',
      consensus: 'Consensus',
      dimension: 'Dimension',
      active: 'ACTIVE',
      multiD: 'Multi-D'
    },
    stargate: {
      status: 'Portal Status',
      connecting: 'Initiating consciousness bridge',
      refresh: 'Refresh Portal',
      title: 'Terra Nova Stargate',
      subtitle: 'Cosmic Gateway to Universal Consciousness',
      description: 'Enter the portal to access the multidimensional network'
    },
    dashboard: {
      title: 'ZION Dashboard',
      subtitle: 'Real-time network monitoring • Dharma-tech metrics',
      loading: 'Loading cosmic data…',
      error: 'Error',
      blockHeight: 'Block Height',
      lastBlock: 'Last Block',
      avgInterval: 'Avg Interval',
      blocksHour: 'Blocks/Hour',
      miningTemples: 'Cosmic Mining Temples',
      recentBlocks: 'Recent Blocks',
      height: 'Height',
      hash: 'Hash',
      time: 'Time',
      age: 'Age',
      hashRate: 'HashRate',
      miners: 'Miners',
      lastBlockTime: 'Last Block',
      latestActivity: 'Latest Blockchain Activity',
      totalMined: 'Total Mined'
    },
    footer: {
      dna: 'DNA Activation Protocol • Dimensional Bridge Technology',
      vzestup: 'Vzestup ∞ • Portugal → ZION Connection • New Earth Origins',
      version: 'ZION v2.5 TestNet • Cosmic Dharma Blockchain'
    },
    temple: {
      '4000d': 'Temple of 4000D',
      '888d': 'Temple of 888D',
      '777d': 'Temple of 777D'
    },
    dimension: {
      transcendent: 'Transcendent Realm',
      christ: 'Christ Consciousness',
      spiritual: 'Spiritual Mastery'
    }
  },
  cs: {
    nav: {
      networkCore: 'Síťové Jádro',
      dharmaTools: 'Dharma Nástroje',
      cosmicPortal: 'Kosmický Portál',
      dashboard: 'Dashboard',
      explorer: 'Průzkumník',
      genesisHub: 'Genesis Hub',
      wallet: 'Peněženka',
      miner: 'Těžař',
      amenti: 'Amenti Knihovna',
      blog: 'Genesis Blog',
      terranova: 'TerraNova Hub',
      halls: 'Síně Amenti',
      stargate: 'Terra Nova Stargate',
      onelove: 'ONE LOVE',
      languageOfLight: 'Jazyk Světla',
      ekam: 'EKAM Chrám'
    },
    home: {
      title: 'ZION',
      subtitle: 'Kosmický Dharma Blockchain',
      description: 'TerraNova® evoluZion v2 • Multidimenzionální Síť',
      networkStatus: 'Stav Sítě',
      consensus: 'Konsensus',
      dimension: 'Dimenze',
      active: 'AKTIVNÍ',
      multiD: 'Multi-D'
    },
    stargate: {
      status: 'Stav Portálu',
      connecting: 'Spouštím most vědomí',
      refresh: 'Obnovit Portál',
      title: 'Terra Nova Stargate',
      subtitle: 'Kosmická Brána do Univerzálního Vědomí',
      description: 'Vstupte do portálu pro přístup k multidimenzionální síti'
    },
    dashboard: {
      title: 'ZION Dashboard',
      subtitle: 'Monitorování sítě v reálném čase • Dharma-tech metriky',
      loading: 'Načítám kosmická data…',
      error: 'Chyba',
      blockHeight: 'Výška Bloku',
      lastBlock: 'Poslední Blok',
      avgInterval: 'Průměrný Interval',
      blocksHour: 'Bloků za Hodinu',
      miningTemples: 'Kosmické Těžební Chrámy',
      recentBlocks: 'Nedávné Bloky',
      height: 'Výška',
      hash: 'Hash',
      time: 'Čas',
      age: 'Stáří',
      hashRate: 'HashRate',
      miners: 'Těžaři',
      lastBlockTime: 'Poslední Blok',
      latestActivity: 'Nejnovější Blockchain Aktivita',
      totalMined: 'Celkem Vytěženo'
    },
    footer: {
      dna: 'DNA Aktivační Protokol • Technologie Dimenzionálních Mostů',
      vzestup: 'Vzestup ∞ • Portugalsko → ZION Spojení • Nové Země Počátky',
      version: 'ZION v2.5 TestNet • Kosmický Dharma Blockchain'
    },
    temple: {
      '4000d': 'Chrám 4000D',
      '888d': 'Chrám 888D',
      '777d': 'Chrám 777D'
    },
    dimension: {
      transcendent: 'Transcendentní Říše',
      christ: 'Kristovo Vědomí',
      spiritual: 'Duchovní Mistrovství'
    }
  },
  pt: {
    nav: {
      networkCore: 'Núcleo da Rede',
      dharmaTools: 'Ferramentas Dharma',
      cosmicPortal: 'Portal Cósmico',
      dashboard: 'Dashboard',
      explorer: 'Explorador',
      genesisHub: 'Genesis Hub',
      wallet: 'Carteira',
      miner: 'Minerador',
      amenti: 'Biblioteca Amenti',
      blog: 'Genesis Blog',
      terranova: 'TerraNova Hub',
      halls: 'Salões de Amenti',
      stargate: 'Terra Nova Stargate',
      onelove: 'ONE LOVE',
      languageOfLight: 'Linguagem da Luz',
      ekam: 'Templo EKAM'
    },
    home: {
      title: 'ZION',
      subtitle: 'Blockchain Dharma Cósmico',
      description: 'TerraNova® evoluZion v2 • Rede Multidimensional',
      networkStatus: 'Estado da Rede',
      consensus: 'Consenso',
      dimension: 'Dimensão',
      active: 'ATIVO',
      multiD: 'Multi-D'
    },
    stargate: {
      status: 'Estado do Portal',
      connecting: 'Iniciando ponte de consciência',
      refresh: 'Atualizar Portal',
      title: 'Terra Nova Stargate',
      subtitle: 'Portal Cósmico para a Consciência Universal',
      description: 'Entre no portal para acessar a rede multidimensional'
    },
    dashboard: {
      title: 'ZION Dashboard',
      subtitle: 'Monitoramento em tempo real • Métricas dharma-tech',
      loading: 'Carregando dados cósmicos…',
      error: 'Erro',
      blockHeight: 'Altura do Bloco',
      lastBlock: 'Último Bloco',
      avgInterval: 'Intervalo Médio',
      blocksHour: 'Blocos/Hora',
      miningTemples: 'Templos de Mineração Cósmica',
      recentBlocks: 'Blocos Recentes',
      height: 'Altura',
      hash: 'Hash',
      time: 'Tempo',
      age: 'Idade',
      hashRate: 'HashRate',
      miners: 'Mineradores',
      lastBlockTime: 'Último Bloco',
      latestActivity: 'Atividade Blockchain Mais Recente',
      totalMined: 'Total Minerado'
    },
    footer: {
      dna: 'Protocolo de Ativação DNA • Tecnologia Ponte Dimensional',
      vzestup: 'Vzestup ∞ • Portugal → Conexão ZION • Origens da Nova Terra',
      version: 'ZION v2.5 TestNet • Blockchain Dharma Cósmico'
    },
    temple: {
      '4000d': 'Templo de 4000D',
      '888d': 'Templo de 888D',
      '777d': 'Templo de 777D'
    },
    dimension: {
      transcendent: 'Reino Transcendente',
      christ: 'Consciência Crística',
      spiritual: 'Maestria Espiritual'
    }
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');

  // Load language from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('zion-language');
    if (saved && ['en', 'cs', 'pt'].includes(saved)) {
      setLanguage(saved as Language);
    }
  }, []);

  // Save language to localStorage
  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem('zion-language', lang);
  };

  const t = translations[language];

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}

export function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="flex items-center space-x-2 bg-black/30 backdrop-blur-sm rounded-lg border border-purple-500/30 p-2">
      {(['en', 'cs', 'pt'] as Language[]).map((lang) => (
        <button
          key={lang}
          onClick={() => setLanguage(lang)}
          className={`px-3 py-1 rounded text-sm font-semibold transition-all ${
            language === lang
              ? 'bg-purple-500 text-white'
              : 'text-purple-300 hover:text-white hover:bg-purple-700/50'
          }`}
        >
          {lang === 'en' ? 'ENG' : lang === 'cs' ? 'CZ' : 'PTG'}
        </button>
      ))}
    </div>
  );
}