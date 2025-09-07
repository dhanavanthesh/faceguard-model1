import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import { 
  Layout, 
  Menu, 
  Typography, 
  Badge, 
  Alert, 
  Spin,
  Button,
  Space,
  notification
} from 'antd';
import {
  DashboardOutlined,
  VideoCameraOutlined,
  UserAddOutlined,
  TeamOutlined,
  SettingOutlined,
  HeartOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

// Components
import Dashboard from './components/Dashboard/Dashboard';
import LiveRecognition from './components/LiveRecognition/LiveRecognition';
import Registration from './components/Registration/Registration';
import UserManagement from './components/UserManagement/UserManagement';

// Services
import apiService from './services/api';

const { Header, Sider, Content, Footer } = Layout;
const { Title, Text } = Typography;

// Main App Component (inside Router)
const AppContent = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State management
  const [collapsed, setCollapsed] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [systemError, setSystemError] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check system health on mount
  useEffect(() => {
    checkSystemHealth();
    
    // Set up periodic health checks
    const healthCheckInterval = setInterval(checkSystemHealth, 60000); // Every minute
    
    return () => clearInterval(healthCheckInterval);
  }, []);

  // Get current path from location
  const currentPath = location.pathname === '/' ? '/dashboard' : location.pathname;

  const checkSystemHealth = async () => {
    try {
      const [healthResponse, statusResponse] = await Promise.all([
        apiService.healthCheck(),
        apiService.getSystemStatus()
      ]);

      setSystemStatus({
        health: healthResponse,
        status: statusResponse
      });
      
      setSystemError(null);
      
      // Show notification for system issues
      if (!healthResponse.services?.face_recognition || !healthResponse.services?.database) {
        notification.warning({
          message: 'System Warning',
          description: 'Some system components may not be fully operational',
          placement: 'topRight'
        });
      }
      
    } catch (error) {
      console.error('System health check failed:', error);
      setSystemError(error.message || 'System health check failed');
      
      notification.error({
        message: 'System Error',
        description: 'Unable to connect to the backend system',
        placement: 'topRight'
      });
    } finally {
      setLoading(false);
    }
  };

  // Menu items configuration
  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
      title: 'System Overview'
    },
    {
      key: '/live-recognition',
      icon: <VideoCameraOutlined />,
      label: 'Live Recognition',
      title: 'Real-time Face Recognition'
    },
    {
      key: '/registration',
      icon: <UserAddOutlined />,
      label: 'Register Person',
      title: 'Add New Person'
    },
    {
      key: '/user-management',
      icon: <TeamOutlined />,
      label: 'User Management',
      title: 'Manage Registered Users'
    }
  ];

  // Handle menu selection
  const handleMenuSelect = ({ key }) => {
    navigate(key);
  };

  // Get current page title
  const getCurrentPageTitle = () => {
    const currentItem = menuItems.find(item => item.key === currentPath);
    return currentItem ? currentItem.title : 'Face Recognition System';
  };

  // Get system status indicator
  const getSystemStatusIndicator = () => {
    if (!systemStatus) {
      return { status: 'default', text: 'Unknown' };
    }

    const { health, status } = systemStatus;
    
    if (health.status === 'healthy' && status.system_ready) {
      return { status: 'success', text: 'Operational' };
    } else if (health.status === 'healthy') {
      return { status: 'processing', text: 'Initializing' };
    } else {
      return { status: 'error', text: 'Error' };
    }
  };

  const statusIndicator = getSystemStatusIndicator();

  if (loading) {
    return (
      <div style={{ 
        height: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        flexDirection: 'column'
      }}>
        <Spin size="large" />
        <Text style={{ marginTop: '16px' }}>
          Connecting to Face Recognition System...
        </Text>
      </div>
    );
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* Sidebar */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="dark"
        width={250}
      >
        {/* Logo/Brand */}
        <div style={{ 
          height: '64px', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: collapsed ? 'center' : 'flex-start',
          padding: collapsed ? '0' : '0 24px',
          background: 'rgba(255, 255, 255, 0.1)'
        }}>
          {!collapsed ? (
            <Title level={4} style={{ color: 'white', margin: 0 }}>
               Face Recognition
            </Title>
          ) : (
            <Text style={{ color: 'white', fontSize: '20px' }}>
              
            </Text>
          )}
        </div>

        {/* Navigation Menu */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[currentPath]}
          items={menuItems}
          onSelect={handleMenuSelect}
          style={{ borderRight: 0 }}
        />

        {/* System Status in Sidebar (when not collapsed) */}
        {!collapsed && (
          <div style={{ 
            position: 'absolute', 
            bottom: '80px', 
            left: '16px', 
            right: '16px',
            padding: '12px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '4px'
          }}>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text style={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '12px' }}>
                  System Status
                </Text>
                <Badge status={statusIndicator.status} />
              </div>
              <Text style={{ color: 'white', fontSize: '12px' }}>
                {statusIndicator.text}
              </Text>
              {systemStatus?.status && (
                <Text style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '10px' }}>
                  Users: {systemStatus.status.total_users || 0} | 
                  DB: {systemStatus.status.face_database_size || 0}
                </Text>
              )}
            </Space>
          </div>
        )}
      </Sider>

      {/* Main Content Area */}
      <Layout>
        {/* Header */}
        <Header style={{ 
          background: '#fff', 
          padding: '0 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 1px 4px rgba(0,21,41,.08)'
        }}>
          <div>
            <Title level={3} style={{ margin: 0 }}>
              {getCurrentPageTitle()}
            </Title>
          </div>
          
          <Space>
            <Badge status={statusIndicator.status} text={statusIndicator.text} />
            <Button 
              type="text" 
              icon={<HeartOutlined />}
              onClick={checkSystemHealth}
              title="Check System Health"
            />
          </Space>
        </Header>

        {/* Content */}
        <Content style={{ background: '#f0f2f5' }}>
          {/* System Error Alert */}
          {systemError && (
            <Alert
              message="System Connection Error"
              description={
                <div>
                  <p>{systemError}</p>
                  <Button size="small" onClick={checkSystemHealth}>
                    Retry Connection
                  </Button>
                </div>
              }
              type="error"
              showIcon
              icon={<ExclamationCircleOutlined />}
              closable
              onClose={() => setSystemError(null)}
              style={{ margin: '16px 24px 0 24px' }}
            />
          )}

          {/* System Not Ready Warning */}
          {systemStatus && !systemStatus.status?.system_ready && !systemError && (
            <Alert
              message="System Initializing"
              description="The face recognition system is starting up. Some features may not be available yet."
              type="warning"
              showIcon
              style={{ margin: '16px 24px 0 24px' }}
            />
          )}

          {/* Route Components */}
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/live-recognition" element={<LiveRecognition />} />
            <Route path="/registration" element={<Registration />} />
            <Route path="/user-management" element={<UserManagement />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Content>

        {/* Footer */}
        <Footer style={{ 
          textAlign: 'center',
          background: '#fff',
          borderTop: '1px solid #f0f0f0'
        }}>
          <Space split={<span style={{ color: '#d9d9d9' }}>|</span>}>
            <Text type="secondary">
              Face Recognition System © 2024
            </Text>
            <Text type="secondary">
              Powered by InsightFace & FastAPI
            </Text>
            <Text type="secondary">
              Built with React & Ant Design
            </Text>
          </Space>
        </Footer>
      </Layout>
    </Layout>
  );
};

// App wrapper with Router
function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;