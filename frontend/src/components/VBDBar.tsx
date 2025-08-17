interface VBDBarProps {
  value: number
  max: number
  delta?: number
}

export function VBDBar({ value, max, delta }: VBDBarProps) {
  const percentage = Math.min((value / max) * 100, 100)
  
  // Color based on value
  const getBarColor = () => {
    if (value > 100) return 'from-green-500 to-emerald-600'
    if (value > 50) return 'from-blue-500 to-indigo-600'
    if (value > 0) return 'from-yellow-500 to-orange-600'
    return 'from-gray-400 to-gray-500'
  }
  
  return (
    <div className="relative h-6 w-full bg-gray-100 rounded-md overflow-hidden">
      <div 
        className={`absolute h-full bg-gradient-to-r ${getBarColor()} transition-all duration-300`}
        style={{ width: `${percentage}%` }}
      />
      {delta && delta !== 0 && (
        <div className="absolute right-2 top-0 h-full flex items-center">
          <span className={`text-xs font-medium ${
            delta > 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {delta > 0 ? '+' : ''}{delta.toFixed(1)}
          </span>
        </div>
      )}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-medium text-gray-700">
          {value.toFixed(1)}
        </span>
      </div>
    </div>
  )
}