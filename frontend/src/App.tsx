import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DraftProvider } from './contexts/DraftContext'
import { Layout } from './components/Layout'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Refetch every 5 seconds for real-time updates
      staleTime: 2000,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DraftProvider>
        <div className="min-h-screen bg-gray-50">
          <Layout />
        </div>
      </DraftProvider>
    </QueryClientProvider>
  )
}

export default App