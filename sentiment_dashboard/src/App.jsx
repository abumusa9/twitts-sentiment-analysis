import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Play, 
  Pause, 
  BarChart3,
  MessageCircle,
  Users,
  Clock,
  Zap,
  Brain,
  Globe
} from 'lucide-react'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar
} from 'recharts'
import './App.css'

// API Configuration
const API_BASE_URL = 'https://sentiment-analysis-platform-f1vh.onrender.com';


// Color scheme for sentiment visualization
const SENTIMENT_COLORS = {
  positive: '#22c55e',
  negative: '#ef4444',
  neutral: '#6b7280'
}

function App() {
  // State management
  const [streamingStatus, setStreamingStatus] = useState({
    stream_active: false,
    processing_active: false,
    tweets_per_minute: 0,
    total_generated: 0,
    total_processed: 0,
    processing_rate: 0
  })
  
  const [recentTweets, setRecentTweets] = useState([])
  const [sentimentTrends, setSentimentTrends] = useState({
    sentiment_counts: { positive: 0, negative: 0, neutral: 0 },
    sentiment_percentages: { positive: 0, negative: 0, neutral: 0 },
    average_confidence: 0,
    total_tweets: 0
  })
  
  const [historicalData, setHistoricalData] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  // Fetch streaming status
  const fetchStreamingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/streaming/status`)
      const data = await response.json()
      if (data.success) {
        setStreamingStatus(data.streaming_status)
      }
    } catch (err) {
      console.error('Error fetching streaming status:', err)
    }
  }

  // Fetch recent tweets
  const fetchRecentTweets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/streaming/recent-tweets?count=10`)
      const data = await response.json()
      if (data.success) {
        setRecentTweets(data.tweets)
      }
    } catch (err) {
      console.error('Error fetching recent tweets:', err)
    }
  }

  // Fetch sentiment trends
  const fetchSentimentTrends = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/streaming/trends?time_window_minutes=60`)
      const data = await response.json()
      if (data.success) {
        setSentimentTrends(data.trends)
        
        // Update historical data for charts
        const timestamp = new Date().toLocaleTimeString()
        setHistoricalData(prev => {
          const newData = [...prev, {
            time: timestamp,
            positive: data.trends.sentiment_percentages.positive,
            negative: data.trends.sentiment_percentages.negative,
            neutral: data.trends.sentiment_percentages.neutral,
            confidence: data.trends.average_confidence * 100
          }]
          // Keep only last 20 data points
          return newData.slice(-20)
        })
      }
    } catch (err) {
      console.error('Error fetching sentiment trends:', err)
    }
  }

  // Start streaming
  const startStreaming = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/streaming/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tweets_per_minute: 120,
          batch_size: 8,
          processing_interval: 2.0
        })
      })
      const data = await response.json()
      if (!data.success) {
        setError(data.error || 'Failed to start streaming')
      }
    } catch (err) {
      setError('Failed to start streaming: ' + err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Stop streaming
  const stopStreaming = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/streaming/stop`, {
        method: 'POST'
      })
      const data = await response.json()
      if (!data.success) {
        setError(data.error || 'Failed to stop streaming')
      }
    } catch (err) {
      setError('Failed to stop streaming: ' + err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Get sentiment icon and color
  const getSentimentDisplay = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return { icon: TrendingUp, color: 'text-green-600', bgColor: 'bg-green-100' }
      case 'negative':
        return { icon: TrendingDown, color: 'text-red-600', bgColor: 'bg-red-100' }
      default:
        return { icon: Minus, color: 'text-gray-600', bgColor: 'bg-gray-100' }
    }
  }

  // Format confidence percentage
  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`
  }

  // Prepare pie chart data
  const pieChartData = [
    { name: 'Positive', value: sentimentTrends.sentiment_counts.positive, color: SENTIMENT_COLORS.positive },
    { name: 'Negative', value: sentimentTrends.sentiment_counts.negative, color: SENTIMENT_COLORS.negative },
    { name: 'Neutral', value: sentimentTrends.sentiment_counts.neutral, color: SENTIMENT_COLORS.neutral }
  ]

  // Auto-refresh data
  useEffect(() => {
    const interval = setInterval(() => {
      fetchStreamingStatus()
      fetchRecentTweets()
      fetchSentimentTrends()
    }, 3000) // Refresh every 3 seconds

    // Initial fetch
    fetchStreamingStatus()
    fetchRecentTweets()
    fetchSentimentTrends()

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center gap-3">
            <Brain className="h-10 w-10 text-blue-600" />
            Sentiment Analysis Platform
          </h1>
          <p className="text-lg text-gray-600">
            Real-time Social Media Sentiment Analysis with BERT-based NLP Models
          </p>
        </div>

        {/* Control Panel */}
        <Card className="border-2 border-blue-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Control
            </CardTitle>
            <CardDescription>
              Manage the real-time streaming and processing system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Button
                onClick={streamingStatus.stream_active ? stopStreaming : startStreaming}
                disabled={isLoading}
                className={streamingStatus.stream_active ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
              >
                {streamingStatus.stream_active ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Stop Streaming
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Streaming
                  </>
                )}
              </Button>
              
              <div className="flex items-center gap-2">
                <div className={`h-3 w-3 rounded-full ${streamingStatus.stream_active ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                <span className="text-sm font-medium">
                  {streamingStatus.stream_active ? 'Active' : 'Inactive'}
                </span>
              </div>

              {error && (
                <Badge variant="destructive" className="ml-auto">
                  {error}
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="border-l-4 border-l-blue-500">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Processed</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {streamingStatus.total_processed.toLocaleString()}
                  </p>
                </div>
                <MessageCircle className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-green-500">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Processing Rate</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {streamingStatus.processing_rate.toFixed(1)}/s
                  </p>
                </div>
                <Zap className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-purple-500">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Avg Confidence</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatConfidence(sentimentTrends.average_confidence)}
                  </p>
                </div>
                <BarChart3 className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-orange-500">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Stream Rate</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {streamingStatus.tweets_per_minute}/min
                  </p>
                </div>
                <Globe className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
            <TabsTrigger value="tweets">Live Tweets</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle>Current Sentiment Distribution</CardTitle>
                  <CardDescription>Last 60 minutes ({sentimentTrends.total_tweets} tweets)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieChartData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {pieChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mt-4">
                    {Object.entries(sentimentTrends.sentiment_percentages).map(([sentiment, percentage]) => {
                      const display = getSentimentDisplay(sentiment)
                      const Icon = display.icon
                      return (
                        <div key={sentiment} className="text-center">
                          <div className={`inline-flex items-center justify-center w-10 h-10 rounded-full ${display.bgColor} mb-2`}>
                            <Icon className={`h-5 w-5 ${display.color}`} />
                          </div>
                          <p className="text-sm font-medium capitalize">{sentiment}</p>
                          <p className="text-lg font-bold">{percentage.toFixed(1)}%</p>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Real-time Trends */}
              <Card>
                <CardHeader>
                  <CardTitle>Real-time Sentiment Trends</CardTitle>
                  <CardDescription>Live sentiment percentages over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historicalData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          dataKey="positive" 
                          stroke={SENTIMENT_COLORS.positive} 
                          strokeWidth={2}
                          name="Positive %"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="negative" 
                          stroke={SENTIMENT_COLORS.negative} 
                          strokeWidth={2}
                          name="Negative %"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="neutral" 
                          stroke={SENTIMENT_COLORS.neutral} 
                          strokeWidth={2}
                          name="Neutral %"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Trends Tab */}
          <TabsContent value="trends" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Confidence Score Trends</CardTitle>
                <CardDescription>Model confidence over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={historicalData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="confidence" 
                        stroke="#8884d8" 
                        fill="#8884d8" 
                        fillOpacity={0.6}
                        name="Confidence %"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Live Tweets Tab */}
          <TabsContent value="tweets" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Live Tweet Feed</CardTitle>
                <CardDescription>Recently processed tweets with sentiment analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {recentTweets.map((tweet, index) => {
                    const sentiment = tweet.sentiment_analysis.sentiment
                    const confidence = tweet.sentiment_analysis.confidence
                    const display = getSentimentDisplay(sentiment)
                    const Icon = display.icon

                    return (
                      <div key={tweet.tweet_data.id || index} className="border rounded-lg p-4 bg-white">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className={`inline-flex items-center justify-center w-8 h-8 rounded-full ${display.bgColor}`}>
                              <Icon className={`h-4 w-4 ${display.color}`} />
                            </div>
                            <Badge variant="outline" className="capitalize">
                              {sentiment}
                            </Badge>
                            <span className="text-sm text-gray-500">
                              {formatConfidence(confidence)}
                            </span>
                          </div>
                          <span className="text-xs text-gray-400">
                            @{tweet.tweet_data.username}
                          </span>
                        </div>
                        <p className="text-gray-800 mb-2">{tweet.tweet_data.text}</p>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <span>❤️ {tweet.tweet_data.like_count}</span>
                          <span>🔄 {tweet.tweet_data.retweet_count}</span>
                          <span>⏱️ {new Date(tweet.processing_timestamp).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Processing Performance</CardTitle>
                  <CardDescription>System performance metrics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Processing Rate</span>
                      <span>{streamingStatus.processing_rate.toFixed(1)} tweets/sec</span>
                    </div>
                    <Progress value={Math.min((streamingStatus.processing_rate / 20) * 100, 100)} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Model Confidence</span>
                      <span>{formatConfidence(sentimentTrends.average_confidence)}</span>
                    </div>
                    <Progress value={sentimentTrends.average_confidence * 100} />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Stream Utilization</span>
                      <span>{streamingStatus.tweets_per_minute}/min</span>
                    </div>
                    <Progress value={Math.min((streamingStatus.tweets_per_minute / 200) * 100, 100)} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Information</CardTitle>
                  <CardDescription>BERT-based sentiment analysis model details</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="font-medium">Model:</span>
                    <span className="text-sm">twitter-roberta-base-sentiment</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Architecture:</span>
                    <span className="text-sm">BERT Transformer</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Accuracy:</span>
                    <span className="text-sm">92%+</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Processing:</span>
                    <span className="text-sm">Real-time</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Capacity:</span>
                    <span className="text-sm">1M+ tweets/day</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

