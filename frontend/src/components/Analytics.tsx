import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { useQuery } from '@tanstack/react-query'
import { useDraft } from '../contexts/DraftContext'

interface AnalyticsProps {
  selectedPlayer: any
}

export function Analytics({ selectedPlayer }: AnalyticsProps) {
  const { currentPick, nextPick } = useDraft()
  const scatterRef = useRef<SVGSVGElement>(null)
  
  // Fetch all players for scatter plot
  const { data: allPlayers = [] } = useQuery({
    queryKey: ['all-players', currentPick],
    queryFn: async () => {
      const response = await fetch(`/api/players?current_pick=${currentPick}`)
      return response.json()
    },
  })
  
  // Position scarcity data
  const positionCounts = allPlayers.reduce((acc: any, player: any) => {
    acc[player.position] = (acc[player.position] || 0) + 1
    return acc
  }, {})
  
  // Create scatter plot
  useEffect(() => {
    if (!scatterRef.current || allPlayers.length === 0) return
    
    const svg = d3.select(scatterRef.current)
    svg.selectAll("*").remove()
    
    const margin = { top: 20, right: 20, bottom: 40, left: 50 }
    const width = 350 - margin.left - margin.right
    const height = 250 - margin.top - margin.bottom
    
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)
    
    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(allPlayers, (d: any) => d.Custom_VBD) || 150])
      .range([0, width])
    
    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0])
    
    // Axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "black")
      .style("font-size", "12px")
      .text("VBD Score")
    
    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -35)
      .attr("x", -height / 2)
      .attr("fill", "black")
      .style("font-size", "12px")
      .text("Availability %")
    
    // Color scale for positions
    const colorScale = d3.scaleOrdinal()
      .domain(['QB', 'RB', 'WR', 'TE'])
      .range(['#8b5cf6', '#10b981', '#3b82f6', '#f97316'])
    
    // Add dots
    g.selectAll(".dot")
      .data(allPlayers.slice(0, 50)) // Top 50 players
      .enter().append("circle")
      .attr("class", "dot")
      .attr("r", 4)
      .attr("cx", (d: any) => xScale(d.Custom_VBD || 0))
      .attr("cy", (d: any) => yScale(d[`prob_pick_${nextPick}`] || 0))
      .style("fill", (d: any) => colorScale(d.position) as string)
      .style("opacity", 0.7)
      .on("mouseover", function(event: any, d: any) {
        // Tooltip
        const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0, 0, 0, 0.8)")
          .style("color", "white")
          .style("padding", "5px 10px")
          .style("border-radius", "4px")
          .style("font-size", "12px")
          .style("pointer-events", "none")
        
        tooltip.html(`${d.player_name}<br/>VBD: ${d.Custom_VBD?.toFixed(1)}<br/>Avail: ${d[`prob_pick_${nextPick}`]?.toFixed(0)}%`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px")
        
        d3.select(this).attr("r", 6)
      })
      .on("mouseout", function() {
        d3.select("body").selectAll(".tooltip").remove()
        d3.select(this).attr("r", 4)
      })
    
    // Highlight selected player
    if (selectedPlayer) {
      g.append("circle")
        .attr("r", 8)
        .attr("cx", xScale(selectedPlayer.Custom_VBD || 0))
        .attr("cy", yScale(selectedPlayer[`prob_pick_${nextPick}`] || 0))
        .style("fill", "none")
        .style("stroke", "#ef4444")
        .style("stroke-width", 2)
    }
    
  }, [allPlayers, selectedPlayer, nextPick])
  
  return (
    <div className="p-4 space-y-4">
      {/* Risk vs Reward Scatter Plot */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Risk vs Reward</h3>
        <svg ref={scatterRef} width="350" height="250"></svg>
        <div className="mt-2 flex justify-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span>QB</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>RB</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>WR</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span>TE</span>
          </div>
        </div>
      </div>
      
      {/* Position Scarcity */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Position Scarcity</h3>
        <div className="space-y-3">
          {['RB', 'WR', 'QB', 'TE'].map(pos => {
            const count = positionCounts[pos] || 0
            const maxCount = 40
            const percentage = (count / maxCount) * 100
            const isLow = count < 10
            
            return (
              <div key={pos}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium">{pos}</span>
                  <span className={isLow ? 'text-red-600 font-semibold' : 'text-gray-600'}>
                    {count} left
                  </span>
                </div>
                <div className="h-6 bg-gray-100 rounded-full overflow-hidden relative">
                  <div 
                    className={`h-full transition-all ${
                      isLow ? 'bg-gradient-to-r from-red-500 to-orange-500' :
                      percentage > 50 ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                      'bg-gradient-to-r from-yellow-500 to-orange-500'
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                  {/* Thermometer effect */}
                  <div className="absolute inset-0 flex items-center px-2">
                    <div className="flex gap-1">
                      {[...Array(5)].map((_, i) => (
                        <div 
                          key={i}
                          className={`h-3 w-1 ${
                            i < Math.ceil(percentage / 20) ? 'bg-white/30' : 'bg-gray-300'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
      
      {/* Quick Picks */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Quick Picks</h3>
        <div className="grid grid-cols-2 gap-2">
          <button className="px-3 py-2 bg-green-100 text-green-700 rounded-md text-sm font-medium hover:bg-green-200">
            Best Available
          </button>
          <button className="px-3 py-2 bg-blue-100 text-blue-700 rounded-md text-sm font-medium hover:bg-blue-200">
            Best RB
          </button>
          <button className="px-3 py-2 bg-purple-100 text-purple-700 rounded-md text-sm font-medium hover:bg-purple-200">
            Best WR
          </button>
          <button className="px-3 py-2 bg-orange-100 text-orange-700 rounded-md text-sm font-medium hover:bg-orange-200">
            Safest Pick
          </button>
        </div>
      </div>
      
      {/* Selected Player Details */}
      {selectedPlayer && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-sm font-semibold text-gray-600 mb-3">
            {selectedPlayer.player_name}
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Position</span>
              <span className="font-medium">{selectedPlayer.position} - {selectedPlayer.team}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">VBD Score</span>
              <span className="font-medium">{selectedPlayer.Custom_VBD?.toFixed(1)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">ESPN Rank</span>
              <span className="font-medium">#{selectedPlayer.overall_rank}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Availability</span>
              <span className={`font-medium ${
                (selectedPlayer[`prob_pick_${nextPick}`] || 0) > 70 ? 'text-green-600' :
                (selectedPlayer[`prob_pick_${nextPick}`] || 0) > 30 ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {(selectedPlayer[`prob_pick_${nextPick}`] || 0).toFixed(0)}%
              </span>
            </div>
            <div className="pt-2 border-t">
              <button 
                className="w-full px-3 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
                onClick={() => {
                  // Simulate pick action
                  fetch('/api/draft-player', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ player_name: selectedPlayer.player_name })
                  })
                }}
              >
                Draft {selectedPlayer.player_name}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}