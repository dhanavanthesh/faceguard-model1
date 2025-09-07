import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { ConfigProvider } from 'antd';
import './index.css';

// Ant Design theme configuration
const theme = {
  token: {
    // Primary colors
    colorPrimary: '#1890ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#f5222d',
    colorInfo: '#1890ff',
    
    // Layout colors
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',
    colorBgLayout: '#f0f2f5',
    
    // Border radius
    borderRadius: 6,
    borderRadiusLG: 8,
    borderRadiusSM: 4,
    
    // Font settings
    fontSize: 14,
    fontSizeLG: 16,
    fontSizeSM: 12,
    
    // Spacing
    padding: 16,
    paddingLG: 24,
    paddingSM: 12,
    
    // Box shadow
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
    boxShadowSecondary: '0 4px 12px rgba(0, 0, 0, 0.15)',
  },
  components: {
    // Layout customization
    Layout: {
      headerBg: '#ffffff',
      siderBg: '#001529',
      bodyBg: '#f0f2f5',
    },
    
    // Menu customization
    Menu: {
      itemBg: 'transparent',
      itemSelectedBg: '#1890ff',
      itemHoverBg: 'rgba(255, 255, 255, 0.1)',
      itemSelectedColor: '#ffffff',
      itemColor: 'rgba(255, 255, 255, 0.65)',
    },
    
    // Card customization
    Card: {
      headerBg: '#fafafa',
      borderRadius: 8,
    },
    
    // Button customization
    Button: {
      borderRadius: 6,
    },
    
    // Table customization
    Table: {
      headerBg: '#fafafa',
      borderRadius: 6,
    }
  }
};

// Create root element
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the app
root.render(
  <React.StrictMode>
    <ConfigProvider theme={theme}>
      <App />
    </ConfigProvider>
  </React.StrictMode>
);