import { useState } from 'react'
import { PlayerTable } from './PlayerTable'
import { PickContext } from './PickContext'
import { Analytics } from './Analytics'
import { useDraft } from '../contexts/DraftContext'

export function Layout() {
  const { currentPick, nextPick, picksUntilNext } = useDraft()
  const [selectedPlayer, setSelectedPlayer] = useState<any>(null)
  const [positionFilter, setPositionFilter] = useState<string>('ALL')

  return (
    <div className="h-screen overflow-hidden">
      {/* Header */}
      <header className="bg-white border-b px-6 py-3">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">
            Fantasy Football Draft Interface
          </h1>
          <div className="flex items-center gap-4 text-sm">
            <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full">
              Pick #{currentPick}
            </span>
            {nextPick && (
              <span className="text-gray-600">
                Your next: #{nextPick} ({picksUntilNext} picks away)
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Three Column Layout */}
      <div className="flex h-[calc(100vh-60px)]">
        {/* Left Column - Pick Context */}
        <div className="w-80 bg-gray-50 border-r overflow-y-auto">
          <PickContext />
        </div>

        {/* Center Column - Player Workspace */}
        <div className="flex-1 bg-white overflow-hidden flex flex-col">
          {/* Filter Bar */}
          <div className="px-6 py-3 border-b">
            <div className="flex items-center gap-2">
              {['ALL', 'QB', 'RB', 'WR', 'TE'].map(pos => (
                <button
                  key={pos}
                  onClick={() => setPositionFilter(pos)}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                    positionFilter === pos
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {pos}
                </button>
              ))}
            </div>
          </div>

          {/* Player Table */}
          <div className="flex-1 overflow-auto">
            <PlayerTable 
              positionFilter={positionFilter}
              onPlayerSelect={setSelectedPlayer}
            />
          </div>
        </div>

        {/* Right Column - Analytics */}
        <div className="w-96 bg-gray-50 border-l overflow-y-auto">
          <Analytics selectedPlayer={selectedPlayer} />
        </div>
      </div>
    </div>
  )
}