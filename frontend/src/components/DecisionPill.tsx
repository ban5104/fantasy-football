interface DecisionPillProps {
  score: number
  max?: number
}

export function DecisionPill({ score, max = 100 }: DecisionPillProps) {
  // Normalize score to 0-100 range
  const normalizedScore = Math.min(Math.max(score, 0), max)
  const percentage = (normalizedScore / max) * 100
  
  // Determine color and label based on score
  const getConfig = () => {
    if (percentage > 80) return { 
      color: 'bg-green-500', 
      textColor: 'text-white',
      label: 'STRONG'
    }
    if (percentage > 60) return { 
      color: 'bg-emerald-500', 
      textColor: 'text-white',
      label: 'GOOD'
    }
    if (percentage > 40) return { 
      color: 'bg-yellow-500', 
      textColor: 'text-white',
      label: 'FAIR'
    }
    if (percentage > 20) return { 
      color: 'bg-orange-500', 
      textColor: 'text-white',
      label: 'WEAK'
    }
    return { 
      color: 'bg-red-500', 
      textColor: 'text-white',
      label: 'POOR'
    }
  }
  
  const config = getConfig()
  
  return (
    <div className="inline-flex items-center gap-2">
      {/* Gauge visualization */}
      <div className="relative w-16 h-8">
        <svg viewBox="0 0 100 50" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 45 A 35 35 0 0 1 90 45"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="8"
          />
          
          {/* Value arc */}
          <path
            d={`M 10 45 A 35 35 0 0 1 ${10 + (80 * percentage / 100)} 45`}
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className={config.color.replace('bg-', 'text-')}
          />
          
          {/* Center text */}
          <text
            x="50"
            y="40"
            textAnchor="middle"
            className="text-xs font-bold fill-current text-gray-700"
          >
            {normalizedScore.toFixed(0)}
          </text>
        </svg>
      </div>
      
      {/* Label pill */}
      <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${config.color} ${config.textColor}`}>
        {config.label}
      </span>
    </div>
  )
}