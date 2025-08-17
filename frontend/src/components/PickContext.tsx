import { useDraft } from '../contexts/DraftContext'
import { useQuery } from '@tanstack/react-query'

export function PickContext() {
  const { currentPick, nextPick, picksUntilNext, myPicks, draftedPlayers } = useDraft()
  
  // Get top recommendations
  const { data: recommendations = [] } = useQuery({
    queryKey: ['recommendations', currentPick],
    queryFn: async () => {
      const response = await fetch(`/api/players?current_pick=${currentPick}`)
      const players = await response.json()
      
      // Filter and sort for best available
      return players
        .filter((p: any) => !draftedPlayers.has(p.player_name))
        .sort((a: any, b: any) => b.decision_score - a.decision_score)
        .slice(0, 5)
    },
  })
  
  return (
    <div className="p-4 space-y-4">
      {/* Current Pick Card */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">Current Pick</h3>
        <div className="text-3xl font-bold text-blue-600">#{currentPick}</div>
        <div className="mt-2 text-sm text-gray-500">
          Round {Math.ceil(currentPick / 12)}, Pick {((currentPick - 1) % 12) + 1}
        </div>
      </div>
      
      {/* Next Pick Card */}
      {nextPick && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Your Next Pick</h3>
          <div className="text-2xl font-bold text-green-600">#{nextPick}</div>
          <div className="mt-2">
            <div className="text-sm text-gray-500">In {picksUntilNext} selections</div>
            <div className="mt-2 bg-gray-100 rounded-full h-2 overflow-hidden">
              <div 
                className="h-full bg-green-500 transition-all"
                style={{ width: `${((currentPick - 1) / (nextPick - 1)) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
      
      {/* Upcoming Picks */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Your Draft Positions</h3>
        <div className="space-y-2">
          {myPicks.map((pick, idx) => {
            const isPast = pick < currentPick
            const isCurrent = pick === currentPick
            const isNext = pick === nextPick
            
            return (
              <div 
                key={pick}
                className={`flex items-center justify-between py-1 px-2 rounded ${
                  isCurrent ? 'bg-blue-100' :
                  isNext ? 'bg-green-100' :
                  isPast ? 'opacity-50' : ''
                }`}
              >
                <span className="text-sm">
                  Round {idx + 1}
                </span>
                <span className={`font-medium text-sm ${
                  isCurrent ? 'text-blue-600' :
                  isNext ? 'text-green-600' :
                  isPast ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Pick {pick}
                </span>
              </div>
            )
          })}
        </div>
      </div>
      
      {/* Top Recommendations */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Top Available</h3>
        <div className="space-y-2">
          {recommendations.map((player: any, idx: number) => (
            <div key={player.player_name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-gray-400">#{idx + 1}</span>
                <div>
                  <div className="text-sm font-medium">{player.player_name}</div>
                  <div className="text-xs text-gray-500">
                    {player.position} - {player.team}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-blue-600">
                  {player.Custom_VBD?.toFixed(1)}
                </div>
                <div className="text-xs text-gray-500">
                  {(player[`prob_pick_${nextPick}`] || 0).toFixed(0)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Draft History Sparkline */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Draft Pace</h3>
        <div className="h-12 flex items-end justify-between gap-1">
          {Array.from({ length: 12 }, (_, i) => {
            const pickNum = i + 1
            const isPicked = pickNum <= currentPick
            const height = Math.random() * 100 // This would be actual pick value in production
            
            return (
              <div 
                key={i}
                className={`flex-1 ${isPicked ? 'bg-blue-500' : 'bg-gray-300'} rounded-t`}
                style={{ height: `${isPicked ? height : 20}%` }}
                title={`Pick ${pickNum}`}
              />
            )
          })}
        </div>
        <div className="mt-2 text-xs text-gray-500 text-center">
          Last 12 picks
        </div>
      </div>
    </div>
  )
}