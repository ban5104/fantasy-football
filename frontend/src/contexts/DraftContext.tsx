import React, { createContext, useContext, useState, useEffect } from 'react'

interface DraftState {
  currentPick: number
  draftedPlayers: Set<string>
  myPicks: number[]
  nextPick: number | null
  picksUntilNext: number | null
}

interface DraftContextType extends DraftState {
  markDrafted: (playerName: string) => void
  undoDraft: (playerName: string) => void
  setCurrentPick: (pick: number) => void
  advancePick: (count?: number) => void
}

const DraftContext = createContext<DraftContextType | undefined>(undefined)

export function DraftProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<DraftState>({
    currentPick: 1,
    draftedPlayers: new Set(),
    myPicks: [8, 17, 32, 41, 56, 65, 80, 89],
    nextPick: 8,
    picksUntilNext: 7,
  })

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'player_drafted') {
        setState(prev => ({
          ...prev,
          draftedPlayers: new Set([...prev.draftedPlayers, data.player]),
          currentPick: data.current_pick,
        }))
      }
    }

    return () => ws.close()
  }, [])

  // Calculate next pick and picks until
  useEffect(() => {
    const nextPick = state.myPicks.find(p => p > state.currentPick) || null
    const picksUntilNext = nextPick ? nextPick - state.currentPick : null
    
    setState(prev => ({
      ...prev,
      nextPick,
      picksUntilNext,
    }))
  }, [state.currentPick, state.myPicks])

  const markDrafted = async (playerName: string) => {
    const response = await fetch('/api/draft-player', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_name: playerName }),
    })
    
    if (response.ok) {
      const data = await response.json()
      setState(prev => ({
        ...prev,
        draftedPlayers: new Set([...prev.draftedPlayers, playerName]),
        currentPick: data.current_pick,
      }))
    }
  }

  const undoDraft = async (playerName: string) => {
    const response = await fetch('/api/undo-draft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_name: playerName }),
    })
    
    if (response.ok) {
      const data = await response.json()
      const newDrafted = new Set(state.draftedPlayers)
      newDrafted.delete(playerName)
      setState(prev => ({
        ...prev,
        draftedPlayers: newDrafted,
        currentPick: data.current_pick,
      }))
    }
  }

  const setCurrentPick = async (pick: number) => {
    const response = await fetch('/api/set-pick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pick }),
    })
    
    if (response.ok) {
      setState(prev => ({
        ...prev,
        currentPick: pick,
      }))
    }
  }

  const advancePick = (count: number = 1) => {
    setCurrentPick(state.currentPick + count)
  }

  return (
    <DraftContext.Provider
      value={{
        ...state,
        markDrafted,
        undoDraft,
        setCurrentPick,
        advancePick,
      }}
    >
      {children}
    </DraftContext.Provider>
  )
}

export function useDraft() {
  const context = useContext(DraftContext)
  if (context === undefined) {
    throw new Error('useDraft must be used within a DraftProvider')
  }
  return context
}