interface AvailabilityBandProps {
  probabilities: number[]
  picks: number[]
  currentPick: number
}

export function AvailabilityBand({ 
  probabilities, 
  picks, 
  currentPick 
}: AvailabilityBandProps) {
  // Create sparkline path
  const width = 120
  const height = 24
  const padding = 2
  
  // Filter picks that are in the future
  const futurePicks = picks.filter(p => p > currentPick)
  const relevantProbs = probabilities.slice(0, futurePicks.length)
  
  if (relevantProbs.length === 0) return null
  
  // Create points for the sparkline
  const points = relevantProbs.map((prob, i) => {
    const x = (i / (relevantProbs.length - 1 || 1)) * (width - 2 * padding) + padding
    const y = height - (prob / 100) * (height - 2 * padding) - padding
    return `${x},${y}`
  }).join(' ')
  
  // Create gradient based on probability levels
  const getGradientStops = () => {
    return relevantProbs.map((prob, i) => {
      const offset = (i / (relevantProbs.length - 1 || 1)) * 100
      const color = prob > 70 ? '#10b981' : prob > 30 ? '#f59e0b' : '#ef4444'
      return `<stop offset="${offset}%" stop-color="${color}" stop-opacity="0.8"/>`
    }).join('')
  }
  
  return (
    <div className="relative">
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id={`gradient-${picks[0]}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
            <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0.8"/>
          </linearGradient>
        </defs>
        
        {/* Background grid */}
        <line 
          x1={padding} 
          y1={height/2} 
          x2={width-padding} 
          y2={height/2} 
          stroke="#e5e7eb" 
          strokeDasharray="2,2"
        />
        
        {/* Sparkline */}
        <polyline
          points={points}
          fill="none"
          stroke={`url(#gradient-${picks[0]})`}
          strokeWidth="2"
        />
        
        {/* Points */}
        {relevantProbs.map((prob, i) => {
          const x = (i / (relevantProbs.length - 1 || 1)) * (width - 2 * padding) + padding
          const y = height - (prob / 100) * (height - 2 * padding) - padding
          const color = prob > 70 ? '#10b981' : prob > 30 ? '#f59e0b' : '#ef4444'
          
          return (
            <g key={i}>
              <circle 
                cx={x} 
                cy={y} 
                r="3" 
                fill={color}
                className="hover:r-4 transition-all"
              />
              <title>Pick {futurePicks[i]}: {prob.toFixed(0)}%</title>
            </g>
          )
        })}
      </svg>
    </div>
  )
}