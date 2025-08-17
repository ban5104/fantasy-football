import React, { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useDraft } from '../contexts/DraftContext'
import { VBDBar } from './VBDBar'
import { AvailabilityBand } from './AvailabilityBand'
import { DecisionPill } from './DecisionPill'

interface Player {
  player_name: string
  position: string
  team: string
  overall_rank: number
  Custom_VBD: number
  Draft_Rank: number
  decision_score: number
  prob_pick_8?: number
  prob_pick_17?: number
  prob_pick_32?: number
  bye_week?: number
}

interface PlayerTableProps {
  positionFilter: string
  onPlayerSelect: (player: Player) => void
}

export function PlayerTable({ positionFilter, onPlayerSelect }: PlayerTableProps) {
  const { currentPick, myPicks, markDrafted, draftedPlayers } = useDraft()
  const [sortBy, setSortBy] = useState<string>('Draft_Rank')
  const [sortAsc, setSortAsc] = useState<boolean>(true)
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  // Fetch players data
  const { data: players = [], isLoading } = useQuery({
    queryKey: ['players', currentPick, positionFilter],
    queryFn: async () => {
      const response = await fetch(
        `/api/players?current_pick=${currentPick}&position_filter=${positionFilter}`
      )
      return response.json()
    },
  })

  // Sort players
  const sortedPlayers = useMemo(() => {
    const sorted = [...players].sort((a, b) => {
      const aVal = a[sortBy] ?? 999
      const bVal = b[sortBy] ?? 999
      return sortAsc ? aVal - bVal : bVal - aVal
    })
    return sorted
  }, [players, sortBy, sortAsc])

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortAsc(!sortAsc)
    } else {
      setSortBy(column)
      setSortAsc(true)
    }
  }

  const getDecisionNotes = (player: Player) => {
    const nextPick = myPicks.find(p => p > currentPick)
    const probAtNext = player[`prob_pick_${nextPick}`] || 0
    
    if (player.Custom_VBD > 100 && probAtNext < 20) {
      return { text: 'ELITE - DRAFT NOW', color: 'text-red-600' }
    }
    if (probAtNext > 80) {
      return { text: 'SAFE - Can wait', color: 'text-green-600' }
    }
    if (probAtNext > 50) {
      return { text: 'LIKELY - Monitor', color: 'text-yellow-600' }
    }
    return { text: 'RISKY - Consider now', color: 'text-orange-600' }
  }

  if (isLoading) {
    return <div className="flex justify-center py-8">Loading...</div>
  }

  return (
    <table className="w-full">
      <thead className="bg-gray-50 sticky top-0 z-10">
        <tr>
          <th className="px-4 py-2 text-left">
            <button 
              onClick={() => handleSort('Draft_Rank')}
              className="font-semibold text-gray-700 hover:text-gray-900"
            >
              VBD Rank {sortBy === 'Draft_Rank' && (sortAsc ? '↑' : '↓')}
            </button>
          </th>
          <th className="px-4 py-2 text-left">Player</th>
          <th className="px-4 py-2 text-left">Pos</th>
          <th className="px-4 py-2 text-left">Team</th>
          <th className="px-4 py-2 text-left">
            <button 
              onClick={() => handleSort('Custom_VBD')}
              className="font-semibold text-gray-700 hover:text-gray-900"
            >
              VBD Score {sortBy === 'Custom_VBD' && (sortAsc ? '↑' : '↓')}
            </button>
          </th>
          <th className="px-4 py-2 text-left">VBD Visual</th>
          <th className="px-4 py-2 text-left">Availability</th>
          {myPicks.slice(0, 2).map(pick => (
            pick > currentPick && (
              <th key={pick} className="px-4 py-2 text-center">
                <button 
                  onClick={() => handleSort(`prob_pick_${pick}`)}
                  className="font-semibold text-gray-700 hover:text-gray-900"
                >
                  P({pick})% {sortBy === `prob_pick_${pick}` && (sortAsc ? '↑' : '↓')}
                </button>
              </th>
            )
          ))}
          <th className="px-4 py-2 text-center">Decision</th>
          <th className="px-4 py-2 text-center">Actions</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-gray-200">
        {sortedPlayers.map((player) => {
          const isDrafted = draftedPlayers.has(player.player_name)
          const isExpanded = expandedRow === player.player_name
          const decision = getDecisionNotes(player)
          
          return (
            <React.Fragment key={player.player_name}>
              <tr 
                className={`hover:bg-gray-50 transition-colors ${
                  isDrafted ? 'opacity-50 bg-gray-100' : ''
                } ${isExpanded ? 'bg-blue-50' : ''}`}
                onClick={() => !isDrafted && onPlayerSelect(player)}
              >
                <td className="px-4 py-3 font-semibold">
                  {player.Draft_Rank || '-'}
                </td>
                <td className="px-4 py-3">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setExpandedRow(isExpanded ? null : player.player_name)
                    }}
                    className="font-medium text-gray-900 hover:text-blue-600"
                  >
                    {player.player_name}
                  </button>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 text-xs font-medium rounded-md ${
                    player.position === 'RB' ? 'bg-green-100 text-green-700' :
                    player.position === 'WR' ? 'bg-blue-100 text-blue-700' :
                    player.position === 'QB' ? 'bg-purple-100 text-purple-700' :
                    player.position === 'TE' ? 'bg-orange-100 text-orange-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {player.position}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">{player.team}</td>
                <td className="px-4 py-3 font-medium">
                  {player.Custom_VBD?.toFixed(1) || '-'}
                </td>
                <td className="px-4 py-3">
                  <VBDBar value={player.Custom_VBD || 0} max={150} />
                </td>
                <td className="px-4 py-3">
                  <AvailabilityBand 
                    probabilities={[
                      player.prob_pick_8 || 0,
                      player.prob_pick_17 || 0,
                      player.prob_pick_32 || 0,
                    ]}
                    picks={myPicks.slice(0, 3)}
                    currentPick={currentPick}
                  />
                </td>
                {myPicks.slice(0, 2).map(pick => (
                  pick > currentPick && (
                    <td key={pick} className="px-4 py-3 text-center">
                      <span className={`font-medium ${
                        (player[`prob_pick_${pick}`] || 0) > 70 ? 'text-green-600' :
                        (player[`prob_pick_${pick}`] || 0) > 30 ? 'text-yellow-600' :
                        'text-red-600'
                      }`}>
                        {(player[`prob_pick_${pick}`] || 0).toFixed(0)}%
                      </span>
                    </td>
                  )
                ))}
                <td className="px-4 py-3 text-center">
                  <DecisionPill score={player.decision_score || 0} />
                </td>
                <td className="px-4 py-3 text-center">
                  {!isDrafted && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        markDrafted(player.player_name)
                      }}
                      className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700"
                    >
                      Draft
                    </button>
                  )}
                </td>
              </tr>
              
              {/* Expanded Row Details */}
              {isExpanded && !isDrafted && (
                <tr>
                  <td colSpan={11} className="px-4 py-4 bg-blue-50">
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-semibold text-sm text-gray-700 mb-2">
                          Decision Analysis
                        </h4>
                        <p className={`text-sm font-medium ${decision.color}`}>
                          {decision.text}
                        </p>
                        <p className="text-xs text-gray-600 mt-1">
                          Decision Score: {player.decision_score?.toFixed(1)}
                        </p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-sm text-gray-700 mb-2">
                          Probability Breakdown
                        </h4>
                        {myPicks.slice(0, 3).map(pick => (
                          pick > currentPick && (
                            <div key={pick} className="text-xs text-gray-600">
                              Pick {pick}: {(player[`prob_pick_${pick}`] || 0).toFixed(1)}%
                            </div>
                          )
                        ))}
                      </div>
                      <div>
                        <h4 className="font-semibold text-sm text-gray-700 mb-2">
                          Additional Info
                        </h4>
                        <p className="text-xs text-gray-600">
                          ESPN Rank: {player.overall_rank}
                        </p>
                        <p className="text-xs text-gray-600">
                          Bye Week: {player.bye_week || 'N/A'}
                        </p>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          )
        })}
      </tbody>
    </table>
  )
}