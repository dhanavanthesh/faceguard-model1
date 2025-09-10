import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Spin, 
  Alert, 
  Progress, 
  List, 
  Avatar, 
  Button,
  Divider,
  Badge,
  Typography,
  Space
} from 'antd';
import { 
  UserOutlined, 
  EyeOutlined, 
  RobotOutlined, 
  ThunderboltOutlined,
  ReloadOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  InteractionOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  ResponsiveContainer
} from 'recharts';

import apiService from '../../services/api';

const { Title, Text } = Typography;

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [trends, setTrends] = useState([]);
  const [topUsers, setTopUsers] = useState([]);
  const [performance, setPerformance] = useState({});
  const [signStats, setSignStats] = useState(null);
  const [emergencyLogs, setEmergencyLogs] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all dashboard data in parallel
      const promises = [
        apiService.getDashboardStats(),
        apiService.getRecognitionTrends(7),
        apiService.getTopRecognizedUsers(5),
        apiService.getSystemPerformance()
      ];

      // Load sign detection data if available
      try {
        promises.push(apiService.getSignStatistics());
        promises.push(apiService.getEmergencySignLogs(5));
      } catch (signError) {
        console.warn('Sign detection not available:', signError);
      }

      const results = await Promise.allSettled(promises);
      
      // Extract successful results
      const [statsResult, trendsResult, topUsersResult, performanceResult, signStatsResult, emergencyLogsResult] = results;

      if (statsResult.status === 'fulfilled') {
        setDashboardData(statsResult.value);
      }
      if (trendsResult.status === 'fulfilled') {
        setTrends(trendsResult.value.trends || []);
      }
      if (topUsersResult.status === 'fulfilled') {
        setTopUsers(topUsersResult.value.top_users || []);
      }
      if (performanceResult.status === 'fulfilled') {
        setPerformance(performanceResult.value);
      }
      if (signStatsResult && signStatsResult.status === 'fulfilled') {
        setSignStats(signStatsResult.value.statistics);
      }
      if (emergencyLogsResult && emergencyLogsResult.status === 'fulfilled') {
        setEmergencyLogs(emergencyLogsResult.value.logs || []);
      }

    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setError(error.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: '16px' }}>
          <Text>Loading dashboard...</Text>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="Error Loading Dashboard"
        description={error}
        type="error"
        showIcon
        action={
          <Button onClick={loadDashboardData}>
            Retry
          </Button>
        }
      />
    );
  }

  const { 
    basic_stats = {}, 
    recognition_stats = {}, 
    ml_model_stats = {}, 
    user_distribution = {}, 
    recent_activity = [] 
  } = dashboardData || {};

  // Chart colors
  const colors = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'];

  // Format user distribution for pie chart
  const distributionData = Object.entries(user_distribution || {}).map(([key, value], index) => ({
    name: key,
    value: value,
    fill: colors[index % colors.length]
  }));

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            📊 Dashboard
          </Title>
          <Text type="secondary">Face Recognition System Overview</Text>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={handleRefresh}
            loading={refreshing}
          >
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Basic Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Users"
              value={basic_stats?.total_users || 0}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            {basic_stats?.recent_registrations > 0 && (
              <div style={{ marginTop: '8px' }}>
                <Badge 
                  count={basic_stats.recent_registrations} 
                  style={{ backgroundColor: '#52c41a' }}
                />
                <Text type="secondary" style={{ marginLeft: '8px' }}>
                  new this week
                </Text>
              </div>
            )}
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Samples"
              value={basic_stats?.total_samples || 0}
              prefix={<EyeOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                Avg: {basic_stats?.average_confidence?.toFixed(3) || 0} confidence
              </Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Recognition Rate"
              value={recognition_stats?.recognition_rate || 0}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ 
                color: (recognition_stats?.recognition_rate || 0) > 80 ? '#52c41a' : '#faad14' 
              }}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">
                {recognition_stats?.successful_recognitions_30d || 0} successful / {recognition_stats?.total_recognitions_30d || 0} total
              </Text>
            </div>
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="ML Model Status"
              value={ml_model_stats?.is_trained ? "Trained" : "Not Trained"}
              prefix={<RobotOutlined />}
              valueStyle={{ 
                color: ml_model_stats?.is_trained ? '#52c41a' : '#faad14' 
              }}
            />
            {ml_model_stats?.is_training && (
              <div style={{ marginTop: '8px' }}>
                <Badge status="processing" text="Training in progress" />
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* Sign Detection Statistics */}
      {signStats && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={24}>
            <Title level={4}>
              <InteractionOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
              Sign Detection Overview
            </Title>
          </Col>
          
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Gesture Detections"
                value={signStats?.total_detections || 0}
                prefix={<InteractionOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">
                  Avg Confidence: {((signStats?.avg_confidence || 0) * 100).toFixed(1)}%
                </Text>
              </div>
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Emergency Alerts"
                value={signStats?.emergency_detections || 0}
                prefix={<WarningOutlined />}
                valueStyle={{ 
                  color: (signStats?.emergency_detections || 0) > 0 ? '#ff4d4f' : '#52c41a' 
                }}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary">
                  Rate: {(signStats?.emergency_rate || 0).toFixed(1)}%
                </Text>
              </div>
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={12}>
            <Card>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text strong>Most Common Gesture</Text>
                  <div style={{ fontSize: '20px', marginTop: '8px' }}>
                    {signStats?.most_common_gesture && signStats.most_common_gesture !== 'None' 
                      ? signStats.most_common_gesture.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())
                      : 'No gestures detected'
                    }
                  </div>
                </div>
                <InteractionOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* Emergency Gesture Alerts */}
      {emergencyLogs && emergencyLogs.length > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={24}>
            <Card 
              title={
                <Space>
                  <WarningOutlined style={{ color: '#ff4d4f' }} />
                  Recent Emergency Gestures
                </Space>
              }
              size="small"
            >
              <List
                dataSource={emergencyLogs}
                renderItem={(item) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={<Badge status="error" />}
                      title={
                        <Space>
                          <Text strong>
                            {item.gesture.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}
                          </Text>
                          <Text type="secondary">
                            {(item.confidence * 100).toFixed(1)}% confidence
                          </Text>
                        </Space>
                      }
                      description={new Date(item.timestamp).toLocaleString()}
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Charts Row */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {/* Recognition Trends */}
        <Col xs={24} lg={16}>
          <Card title="Recognition Trends (Last 7 Days)" extra={<EyeOutlined />}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="total_recognitions" 
                  stroke="#1890ff" 
                  strokeWidth={2}
                  name="Total Recognitions"
                />
                <Line 
                  type="monotone" 
                  dataKey="successful_recognitions" 
                  stroke="#52c41a" 
                  strokeWidth={2}
                  name="Successful"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* User Distribution */}
        <Col xs={24} lg={8}>
          <Card title="Users by Sample Count" extra={<UserOutlined />}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={distributionData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {distributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Bottom Row */}
      <Row gutter={[16, 16]}>
        {/* System Performance */}
        <Col xs={24} lg={12}>
          <Card title="System Performance" extra={<ThunderboltOutlined />}>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Text strong>Processing Time</Text>
                <div>
                  <Text type="secondary">Average: {performance?.avg_processing_time?.toFixed(4) || 0}s</Text>
                  <Progress 
                    percent={Math.max(0, 100 - (performance?.avg_processing_time || 0) * 100)} 
                    size="small"
                    status={performance?.avg_processing_time < 0.5 ? 'success' : 'normal'}
                    format={() => performance?.performance_grade || 'N/A'}
                  />
                </div>
              </div>

              <div>
                <Text strong>Confidence Level</Text>
                <div>
                  <Text type="secondary">Average: {performance?.avg_confidence?.toFixed(3) || 0}</Text>
                  <Progress 
                    percent={(performance?.avg_confidence || 0) * 100} 
                    size="small"
                    status={performance?.avg_confidence > 0.8 ? 'success' : 'normal'}
                  />
                </div>
              </div>

              <Row gutter={16}>
                <Col span={12}>
                  <Statistic 
                    title="Requests (24h)" 
                    value={performance?.total_requests || 0}
                    size="small"
                  />
                </Col>
                <Col span={12}>
                  <Statistic 
                    title="Performance Grade" 
                    value={performance?.performance_grade || 'N/A'}
                    size="small"
                  />
                </Col>
              </Row>
            </Space>
          </Card>
        </Col>

        {/* Top Recognized Users */}
        <Col xs={24} lg={12}>
          <Card title="Top Recognized Users (30 days)" extra={<TrophyOutlined />}>
            <List
              itemLayout="horizontal"
              dataSource={topUsers}
              renderItem={(user, index) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Badge count={index + 1} style={{ backgroundColor: '#1890ff' }}>
                        <Avatar icon={<UserOutlined />} />
                      </Badge>
                    }
                    title={user.name}
                    description={
                      <Space size="small">
                        <Text type="secondary">
                          {user.recognition_count} recognitions
                        </Text>
                        <Divider type="vertical" />
                        <Text type="secondary">
                          {user.avg_confidence?.toFixed(3)} avg confidence
                        </Text>
                      </Space>
                    }
                  />
                  <div>
                    <Text type="secondary">
                      {user.total_samples} samples
                    </Text>
                  </div>
                </List.Item>
              )}
              locale={{
                emptyText: 'No recognition data available'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="Recent Activity" extra={<ClockCircleOutlined />}>
            <List
              itemLayout="horizontal"
              dataSource={recent_activity?.slice(0, 10) || []}
              renderItem={(activity) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={activity.type === 'registration' ? <UserOutlined /> : <EyeOutlined />}
                        style={{ 
                          backgroundColor: activity.type === 'registration' ? '#52c41a' : '#1890ff' 
                        }}
                      />
                    }
                    title={activity.description}
                    description={
                      <Text type="secondary">
                        {new Date(activity.timestamp).toLocaleString()}
                      </Text>
                    }
                  />
                </List.Item>
              )}
              locale={{
                emptyText: 'No recent activity'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Pre-trained Models Section */}
      <Row style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card 
            title="🚀 Pre-trained Models (Notebook Approach)" 
            extra={<RobotOutlined />}
          >
            <div style={{ marginBottom: '16px' }}>
              <Text>
                This system uses pre-trained models created by <code>training_model.py</code>, 
                similar to your original notebook with enhanced SVM + Random Forest training and mask/unmask detection.
              </Text>
            </div>
            
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Card size="small" title="Training Instructions">
                  <Text type="secondary">
                    Run the training script to create pre-trained models like your notebook:
                  </Text>
                  <div style={{ marginTop: '12px', backgroundColor: '#f6f8fa', padding: '8px', borderRadius: '4px' }}>
                    <Text code>python training_model.py</Text>
                  </div>
                  <div style={{ marginTop: '8px' }}>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      This downloads datasets, trains SVM+RF models, and creates face database
                    </Text>
                  </div>
                </Card>
              </Col>
              
              <Col xs={24} md={12}>
                <Card size="small" title="Model Status">
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={async () => {
                      try {
                        const response = await fetch('http://localhost:8000/api/faces/pretrained-status');
                        const status = await response.json();
                        console.log('Pre-trained models status:', status);
                        setError(null);
                      } catch (error) {
                        setError('Failed to get pre-trained model status');
                      }
                    }}
                  >
                    Check Pre-trained Status
                  </Button>
                  <div style={{ marginTop: '12px' }}>
                    <Text type="secondary">
                      Check console for pre-trained model details
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
            
            <Divider />
            
            <div style={{ backgroundColor: '#e6f7ff', padding: '12px', borderRadius: '4px', border: '1px solid #91d5ff' }}>
              <Text strong>📋 How it works:</Text>
              <ol style={{ marginTop: '8px', marginBottom: '0', paddingLeft: '20px' }}>
                <li><strong>Pre-training:</strong> Run <code>training_model.py</code> once to create models</li>
                <li><strong>Server start:</strong> Models loaded automatically at startup</li>
                <li><strong>Recognition:</strong> Pre-trained models → Runtime database → Cosine similarity</li>
                <li><strong>New users:</strong> Can still register via the web interface</li>
              </ol>
            </div>
            
            <div style={{ backgroundColor: '#f6f8fa', padding: '12px', borderRadius: '4px', marginTop: '12px' }}>
              <Text strong>💡 Training Tips:</Text>
              <ul style={{ marginTop: '8px', marginBottom: '0' }}>
                <li>Add your training images to <code>training_datasets/</code> folders</li>
                <li>Include masked and unmasked variations</li>
                <li>Use diverse lighting and angles</li>
                <li>Run training before starting the server for best results</li>
              </ul>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;