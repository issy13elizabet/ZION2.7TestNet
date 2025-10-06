"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Link from "next/link";

interface StakingPool {
  id: string;
  name: string;
  validator: string;
  apr: number;
  total_staked: number;
  min_stake: number;
  status: 'active' | 'inactive' | 'jailed';
  cosmic_multiplier: number;
  dharma_bonus: number;
  lock_period: number;
}

interface UserStake {
  id: string;
  pool_id: string;
  amount: number;
  rewards_earned: number;
  start_date: string;
  unlock_date: string;
  status: 'active' | 'unbonding' | 'claimable';
  dharma_generated: number;
}

export default function StakingPage() {
  const [stakingPools, setStakingPools] = useState<StakingPool[]>([
    {
      id: 'pool_001',
      name: 'Cosmic Dharma Validator',
      validator: 'zionvaloper1cosmic42dharma108enlightenment21m',
      apr: 21.08,
      total_staked: 4200000,
      min_stake: 100,
      status: 'active',
      cosmic_multiplier: 1.42,
      dharma_bonus: 15,
      lock_period: 21
    },
    {
      id: 'pool_002',
      name: 'Universal Consensus Node',
      validator: 'zionvaloper1universal888consensus216harmony42k',
      apr: 18.96,
      total_staked: 8800000,
      min_stake: 50,
      status: 'active',
      cosmic_multiplier: 1.33,
      dharma_bonus: 12,
      lock_period: 28
    },
    {
      id: 'pool_003',
      name: 'Zen Staking Collective',
      validator: 'zionvaloper1zen108meditation42inner_peace216m',
      apr: 24.42,
      total_staked: 2100000,
      min_stake: 250,
      status: 'active',
      cosmic_multiplier: 1.55,
      dharma_bonus: 20,
      lock_period: 14
    },
    {
      id: 'pool_004',
      name: 'Broken Validator',
      validator: 'zionvaloper1broken404error500downtime666x',
      apr: 0,
      total_staked: 500000,
      min_stake: 10,
      status: 'jailed',
      cosmic_multiplier: 0,
      dharma_bonus: 0,
      lock_period: 30
    }
  ]);

  const [userStakes, setUserStakes] = useState<UserStake[]>([
    {
      id: 'stake_001',
      pool_id: 'pool_001',
      amount: 1000,
      rewards_earned: 42.108,
      start_date: '2024-01-15',
      unlock_date: '2024-02-05',
      status: 'active',
      dharma_generated: 15.5
    },
    {
      id: 'stake_002', 
      pool_id: 'pool_002',
      amount: 500,
      rewards_earned: 21.69,
      start_date: '2024-01-20',
      unlock_date: '2024-02-17',
      status: 'active',
      dharma_generated: 8.2
    }
  ]);

  const [stakeForm, setStakeForm] = useState({
    pool_id: '',
    amount: ''
  });

  const handleStake = async () => {
    if (!stakeForm.pool_id || !stakeForm.amount) {
      alert('Please select a pool and enter amount');
      return;
    }

    const pool = stakingPools.find(p => p.id === stakeForm.pool_id);
    if (!pool) return;

    const amount = parseFloat(stakeForm.amount);
    if (amount < pool.min_stake) {
      alert(`Minimum stake is ${pool.min_stake} ZION`);
      return;
    }

    const newStake: UserStake = {
      id: `stake_${Date.now()}`,
      pool_id: stakeForm.pool_id,
      amount: amount,
      rewards_earned: 0,
      start_date: new Date().toISOString().split('T')[0],
      unlock_date: new Date(Date.now() + pool.lock_period * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      status: 'active',
      dharma_generated: 0
    };

    setUserStakes([...userStakes, newStake]);
    setStakeForm({ pool_id: '', amount: '' });
  };

  const handleUnstake = (stakeId: string) => {
    setUserStakes(stakes => stakes.map(stake => 
      stake.id === stakeId ? { ...stake, status: 'unbonding' as const } : stake
    ));
  };

  const handleClaimRewards = (stakeId: string) => {
    setUserStakes(stakes => stakes.map(stake => {
      if (stake.id === stakeId && stake.rewards_earned > 0) {
        return { ...stake, rewards_earned: 0, dharma_generated: stake.dharma_generated + 5 };
      }
      return stake;
    }));
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-400 bg-green-500/20';
      case 'inactive': return 'text-yellow-400 bg-yellow-500/20';
      case 'jailed': return 'text-red-400 bg-red-500/20';
      case 'unbonding': return 'text-orange-400 bg-orange-500/20';
      case 'claimable': return 'text-blue-400 bg-blue-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'active': return '✅';
      case 'inactive': return '⏸️';
      case 'jailed': return '🚫';
      case 'unbonding': return '⏳';
      case 'claimable': return '💰';
      default: return '❓';
    }
  };

  const getPoolByStake = (stake: UserStake) => {
    return stakingPools.find(pool => pool.id === stake.pool_id);
  };

  const totalStaked = userStakes.filter(s => s.status === 'active').reduce((sum, stake) => sum + stake.amount, 0);
  const totalRewards = userStakes.reduce((sum, stake) => sum + stake.rewards_earned, 0);
  const totalDharma = userStakes.reduce((sum, stake) => sum + stake.dharma_generated, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/wallet" className="inline-block mb-4">
          <motion.button
            className="px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Wallet
          </motion.button>
        </Link>
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent mb-2">
          🥩 Cosmic Staking
        </h1>
        <p className="text-purple-300">Stake ZION & Generate Cosmic Dharma Rewards</p>
      </motion.header>

      {/* Staking Summary */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <motion.div className="bg-gradient-to-br from-purple-500 to-violet-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🥩</div>
            <div className="text-lg font-bold text-violet-300 break-all">{totalStaked.toLocaleString()}</div>
            <div className="text-sm text-violet-200">Total Staked</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-green-500 to-emerald-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">💰</div>
            <div className="text-lg font-bold text-emerald-300 break-all">{totalRewards.toFixed(3)}</div>
            <div className="text-sm text-emerald-200">Pending Rewards</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-pink-500 to-rose-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">☸️</div>
            <div className="text-lg font-bold text-rose-300 break-all">{totalDharma.toFixed(1)}</div>
            <div className="text-sm text-rose-200">Dharma Generated</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-blue-500 to-cyan-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">📈</div>
            <div className="text-2xl font-bold text-cyan-300">
              {userStakes.length > 0 ? 
                (userStakes.map(s => {
                  const pool = getPoolByStake(s);
                  return pool ? pool.apr * pool.cosmic_multiplier : 0;
                }).reduce((sum, apr) => sum + apr, 0) / userStakes.length).toFixed(1) : 
                '0'
              }%
            </div>
            <div className="text-sm text-cyan-200">Avg Cosmic APR</div>
          </div>
        </motion.div>
      </div>

      {/* Stake Form */}
      <motion.div className="mb-8 bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-purple-500/30" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <h2 className="text-xl font-semibold mb-4 text-center text-purple-300">🥩 Stake ZION Tokens</h2>
        
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Select Staking Pool</label>
            <select
              value={stakeForm.pool_id}
              onChange={(e) => setStakeForm({...stakeForm, pool_id: e.target.value})}
              className="w-full px-3 py-2 bg-black/50 border border-purple-500/30 rounded-lg text-white focus:border-purple-400 focus:outline-none"
            >
              <option value="">Choose a validator pool</option>
              {stakingPools.filter(pool => pool.status === 'active').map(pool => (
                <option key={pool.id} value={pool.id}>
                  {pool.name} - {(pool.apr * pool.cosmic_multiplier).toFixed(1)}% APR
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Amount to Stake (ZION)</label>
            <input
              type="number"
              step="0.001"
              value={stakeForm.amount}
              onChange={(e) => setStakeForm({...stakeForm, amount: e.target.value})}
              placeholder="100.000"
              className="w-full px-3 py-2 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:border-purple-400 focus:outline-none"
            />
          </div>
        </div>
        
        {stakeForm.pool_id && stakeForm.amount && (
          <div className="mt-4 p-3 bg-black/30 rounded-lg">
            {(() => {
              const selectedPool = stakingPools.find(p => p.id === stakeForm.pool_id);
              const amount = parseFloat(stakeForm.amount);
              if (!selectedPool || !amount) return null;
              
              const cosmicApr = selectedPool.apr * selectedPool.cosmic_multiplier;
              const dailyRewards = (amount * cosmicApr / 100) / 365;
              
              return (
                <>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Cosmic APR:</span>
                    <span className="text-purple-300">{cosmicApr.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">Daily Rewards:</span>
                    <span className="text-green-300">{dailyRewards.toFixed(6)} ZION</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Lock Period:</span>
                    <span className="text-yellow-300">{selectedPool.lock_period} days</span>
                  </div>
                </>
              );
            })()}
          </div>
        )}
        
        <motion.button
          className="w-full mt-4 bg-gradient-to-r from-purple-500 to-pink-600 px-6 py-3 rounded-xl font-semibold"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleStake}
        >
          🥩 Stake ZION
        </motion.button>
      </motion.div>

      {/* User Stakes */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-cyan-300">🎯 Your Active Stakes</h2>
        
        {userStakes.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-2">🚀</div>
            <p>No active stakes yet. Start staking to earn cosmic rewards!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {userStakes.map((stake, index) => {
              const pool = getPoolByStake(stake);
              if (!pool) return null;
              
              return (
                <motion.div
                  key={stake.id}
                  className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + index * 0.1 }}
                >
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-cyan-300">{pool.name}</h3>
                      <div className="text-sm text-gray-400">Validator: {pool.validator.slice(0, 20)}...</div>
                      <div className="text-sm text-purple-300 mt-1">
                        Staked: {stake.amount.toLocaleString()} ZION
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`px-2 py-1 rounded-lg text-xs font-medium mb-2 ${getStatusColor(stake.status)}`}>
                        {getStatusIcon(stake.status)} {stake.status.toUpperCase()}
                      </div>
                      <div className="text-xs text-gray-400">
                        APR: {(pool.apr * pool.cosmic_multiplier).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-3 mb-4">
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Pending Rewards</div>
                      <div className="text-lg font-bold text-green-300">{stake.rewards_earned.toFixed(6)} ZION</div>
                    </div>
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Dharma Generated</div>
                      <div className="text-lg font-bold text-purple-300">{stake.dharma_generated.toFixed(1)} ☸️</div>
                    </div>
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Unlock Date</div>
                      <div className="text-sm font-bold text-yellow-300">{stake.unlock_date}</div>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    {stake.rewards_earned > 0 && (
                      <motion.button
                        className="flex-1 px-3 py-2 bg-green-600/30 hover:bg-green-600/50 rounded-lg text-sm"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleClaimRewards(stake.id)}
                      >
                        💰 Claim Rewards
                      </motion.button>
                    )}
                    {stake.status === 'active' && (
                      <motion.button
                        className="flex-1 px-3 py-2 bg-orange-600/30 hover:bg-orange-600/50 rounded-lg text-sm"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleUnstake(stake.id)}
                      >
                        ⏳ Unstake
                      </motion.button>
                    )}
                    <motion.button
                      className="flex-1 px-3 py-2 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg text-sm"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      📊 Details
                    </motion.button>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.div>

      {/* Available Staking Pools */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-pink-300">🏊 Available Staking Pools</h2>
        
        <div className="grid gap-4">
          {stakingPools.map((pool, index) => (
            <motion.div
              key={pool.id}
              className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-pink-500/30"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.9 + index * 0.1 }}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-pink-300">{pool.name}</h3>
                  <div className="text-xs text-gray-400 font-mono break-all">{pool.validator}</div>
                </div>
                <div className="text-right">
                  <div className={`px-2 py-1 rounded-lg text-xs font-medium mb-1 ${getStatusColor(pool.status)}`}>
                    {getStatusIcon(pool.status)} {pool.status.toUpperCase()}
                  </div>
                  <div className="text-sm text-purple-300">{pool.cosmic_multiplier}x</div>
                  <div className="text-xs text-purple-200">Cosmic</div>
                </div>
              </div>

              <div className="grid gap-2 md:grid-cols-5 mb-4">
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Base APR</div>
                  <div className="text-sm font-bold text-green-300">{pool.apr}%</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Cosmic APR</div>
                  <div className="text-sm font-bold text-purple-300">{(pool.apr * pool.cosmic_multiplier).toFixed(1)}%</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Total Staked</div>
                  <div className="text-sm font-bold text-blue-300">{pool.total_staked.toLocaleString()}</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Min Stake</div>
                  <div className="text-sm font-bold text-yellow-300">{pool.min_stake}</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Lock Period</div>
                  <div className="text-sm font-bold text-orange-300">{pool.lock_period}d</div>
                </div>
              </div>

              <div className="flex gap-2">
                <motion.button
                  className="flex-1 px-3 py-2 bg-pink-600/30 hover:bg-pink-600/50 rounded-lg text-sm"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  disabled={pool.status !== 'active'}
                  onClick={() => setStakeForm({...stakeForm, pool_id: pool.id})}
                >
                  🥩 Stake Here
                </motion.button>
                <motion.button
                  className="flex-1 px-3 py-2 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg text-sm"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  📊 Pool Stats
                </motion.button>
                <motion.button
                  className="flex-1 px-3 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  ℹ️ Validator Info
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}