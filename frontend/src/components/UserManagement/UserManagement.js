import React, { useState, useEffect } from 'react';
import { 
  Table, 
  Button, 
  Space, 
  Modal, 
  Form,
  Input,
  Card,
  Avatar,
  Row,
  Col,
  Statistic,
  Tag,
  Typography,
  Alert,
  Popconfirm,
  notification,
  Upload,
  Tooltip,
  Badge
} from 'antd';
import { 
  UserOutlined,
  DeleteOutlined,
  PlusOutlined,
  ReloadOutlined,
  CameraOutlined,
  UploadOutlined,
  EyeOutlined,
  WarningOutlined,
  SearchOutlined
} from '@ant-design/icons';

import apiService from '../../services/api';

const { Title, Text } = Typography;
const { Search } = Input;

const UserManagement = () => {
  // State management
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState({});
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredUsers, setFilteredUsers] = useState([]);
  
  // Modal states
  const [addSampleModalVisible, setAddSampleModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [addSampleLoading, setAddSampleLoading] = useState(false);

  // Form instances
  const [addSampleForm] = Form.useForm();

  // Load users on component mount
  useEffect(() => {
    loadUsers();
  }, []);

  // Filter users based on search term
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredUsers(users);
    } else {
      const filtered = users.filter(user => 
        user.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredUsers(filtered);
    }
  }, [users, searchTerm]);

  // Load all users
  const loadUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getAllUsers();
      
      if (response && response.success && Array.isArray(response.users)) {
        setUsers(response.users);
      } else if (Array.isArray(response)) {
        // Fallback for direct array response
        setUsers(response);
      } else {
        console.error('Unexpected response format:', response);
        throw new Error('Invalid response format');
      }
      
    } catch (error) {
      console.error('Error loading users:', error);
      setError(error.message || 'Failed to load users');
      notification.error({
        message: 'Failed to Load Users',
        description: error.message || 'Please try again later'
      });
    } finally {
      setLoading(false);
    }
  };

  // Delete user
  const handleDeleteUser = async (userId, userName) => {
    try {
      setDeleteLoading(prev => ({ ...prev, [userId]: true }));
      
      const response = await apiService.deleteUser(userId);
      
      if (response.success) {
        notification.success({
          message: 'User Deleted',
          description: `${userName} has been successfully deleted`
        });
        
        // Remove user from local state
        setUsers(prev => prev.filter(user => user.id !== userId));
      } else {
        throw new Error(response.message || 'Failed to delete user');
      }
      
    } catch (error) {
      console.error('Error deleting user:', error);
      notification.error({
        message: 'Delete Failed',
        description: error.message || 'Failed to delete user'
      });
    } finally {
      setDeleteLoading(prev => ({ ...prev, [userId]: false }));
    }
  };

  // Add face sample
  const handleAddSample = (user) => {
    setSelectedUser(user);
    setAddSampleModalVisible(true);
    addSampleForm.resetFields();
  };

  // Submit add sample
  const handleAddSampleSubmit = async (values) => {
    try {
      setAddSampleLoading(true);
      
      const { image } = values;
      if (!image || !image[0]) {
        throw new Error('Please select an image file');
      }

      // Convert file to base64
      const base64Image = await apiService.fileToBase64(image[0].originFileObj);
      
      const response = await apiService.addFaceSample(selectedUser.id, base64Image);
      
      if (response.success) {
        notification.success({
          message: 'Sample Added',
          description: `Face sample added for ${selectedUser.name}`
        });
        
        setAddSampleModalVisible(false);
        loadUsers(); // Refresh user list
      } else {
        throw new Error(response.message || 'Failed to add sample');
      }
      
    } catch (error) {
      console.error('Error adding sample:', error);
      notification.error({
        message: 'Add Sample Failed',
        description: error.message || 'Failed to add face sample'
      });
    } finally {
      setAddSampleLoading(false);
    }
  };

  // Calculate statistics
  const totalUsers = users.length;
  const totalSamples = users.reduce((sum, user) => sum + (user.total_samples || 0), 0);
  const avgConfidence = users.length > 0 
    ? users.reduce((sum, user) => sum + (user.confidence_level || 0), 0) / users.length 
    : 0;
  const highConfidenceUsers = users.filter(user => (user.confidence_level || 0) > 0.8).length;

  // Table columns
  const columns = [
    {
      title: 'Avatar',
      dataIndex: 'avatar',
      key: 'avatar',
      width: 60,
      render: (_, record) => (
        <Avatar 
          size="large" 
          icon={<UserOutlined />}
          style={{ backgroundColor: '#1890ff' }}
        >
          {record.name.charAt(0).toUpperCase()}
        </Avatar>
      ),
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (name, record) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            ID: {record.id?.slice(-8) || 'N/A'}
          </Text>
        </div>
      ),
    },
    {
      title: 'Samples',
      dataIndex: 'total_samples',
      key: 'total_samples',
      width: 100,
      sorter: (a, b) => (a.total_samples || 0) - (b.total_samples || 0),
      render: (samples) => (
        <Badge 
          count={samples || 0} 
          style={{ backgroundColor: samples > 5 ? '#52c41a' : '#faad14' }}
          showZero
        />
      ),
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence_level',
      key: 'confidence_level',
      width: 120,
      sorter: (a, b) => (a.confidence_level || 0) - (b.confidence_level || 0),
      render: (confidence) => {
        const value = confidence || 0;
        const color = value > 0.8 ? 'green' : value > 0.6 ? 'orange' : 'red';
        return (
          <Tag color={color}>
            {value.toFixed(3)}
          </Tag>
        );
      },
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      sorter: (a, b) => new Date(a.created_at) - new Date(b.created_at),
      render: (date) => (
        <Text type="secondary">
          {new Date(date).toLocaleDateString()}
        </Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Add Face Sample">
            <Button
              type="primary"
              size="small"
              icon={<CameraOutlined />}
              onClick={() => handleAddSample(record)}
            />
          </Tooltip>
          
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                Modal.info({
                  title: `${record.name} - Details`,
                  width: 500,
                  content: (
                    <div style={{ marginTop: '16px' }}>
                      <p><strong>User ID:</strong> {record.id}</p>
                      <p><strong>Name:</strong> {record.name}</p>
                      <p><strong>Total Samples:</strong> {record.total_samples || 0}</p>
                      <p><strong>Confidence Level:</strong> {(record.confidence_level || 0).toFixed(3)}</p>
                      <p><strong>Created:</strong> {new Date(record.created_at).toLocaleString()}</p>
                      <p><strong>Last Updated:</strong> {new Date(record.updated_at || record.created_at).toLocaleString()}</p>
                    </div>
                  ),
                });
              }}
            />
          </Tooltip>

          <Popconfirm
            title="Delete User"
            description={
              <div>
                <p>Are you sure you want to delete <strong>{record.name}</strong>?</p>
                <p style={{ color: '#ff4d4f', marginBottom: 0 }}>
                  This will permanently remove all face data and cannot be undone.
                </p>
              </div>
            }
            onConfirm={() => handleDeleteUser(record.id, record.name)}
            okText="Delete"
            cancelText="Cancel"
            okType="danger"
            icon={<WarningOutlined style={{ color: '#ff4d4f' }} />}
          >
            <Button
              danger
              size="small"
              icon={<DeleteOutlined />}
              loading={deleteLoading[record.id]}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            👥 User Management
          </Title>
          <Text type="secondary">Manage registered users and face samples</Text>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={loadUsers}
            loading={loading}
          >
            Refresh
          </Button>
        </Col>
      </Row>

      {/* Statistics Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Total Users"
              value={totalUsers}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Total Samples"
              value={totalSamples}
              prefix={<CameraOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="Avg Confidence"
              value={avgConfidence.toFixed(3)}
              prefix={<EyeOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="High Confidence"
              value={highConfidenceUsers}
              suffix={`/ ${totalUsers}`}
              prefix={<Badge status="success" />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Error Loading Users"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* Main Table */}
      <Card 
        title="Registered Users"
        extra={
          <Space>
            <Search
              placeholder="Search users..."
              allowClear
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ width: 200 }}
              prefix={<SearchOutlined />}
            />
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={filteredUsers}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} users`,
          }}
          scroll={{ x: 800 }}
          locale={{
            emptyText: searchTerm 
              ? `No users found matching "${searchTerm}"` 
              : 'No users registered yet'
          }}
        />
      </Card>

      {/* Add Sample Modal */}
      <Modal
        title={`Add Face Sample - ${selectedUser?.name}`}
        open={addSampleModalVisible}
        onCancel={() => {
          setAddSampleModalVisible(false);
          addSampleForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={addSampleForm}
          layout="vertical"
          onFinish={handleAddSampleSubmit}
        >
          <Alert
            message="Adding Face Samples"
            description={
              <div>
                <p>Adding multiple face samples improves recognition accuracy.</p>
                <ul>
                  <li>Use different angles and expressions</li>
                  <li>Ensure good lighting conditions</li>
                  <li>Only one face should be visible in the image</li>
                </ul>
              </div>
            }
            type="info"
            showIcon
            style={{ marginBottom: '16px' }}
          />

          <Form.Item
            name="image"
            label="Face Image"
            rules={[{ required: true, message: 'Please select an image' }]}
            valuePropName="fileList"
            getValueFromEvent={(e) => {
              if (Array.isArray(e)) {
                return e;
              }
              return e && e.fileList;
            }}
          >
            <Upload
              name="image"
              listType="picture-card"
              accept="image/*"
              beforeUpload={() => false}
              maxCount={1}
            >
              <div>
                <UploadOutlined />
                <div style={{ marginTop: 8 }}>Select Image</div>
              </div>
            </Upload>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={addSampleLoading}
                icon={<PlusOutlined />}
              >
                Add Sample
              </Button>
              <Button
                onClick={() => {
                  setAddSampleModalVisible(false);
                  addSampleForm.resetFields();
                }}
              >
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Help Information */}
      {users.length === 0 && !loading && (
        <Card style={{ marginTop: '16px', textAlign: 'center' }}>
          <div style={{ padding: '40px' }}>
            <UserOutlined style={{ fontSize: '64px', color: '#d9d9d9' }} />
            <Title level={4} style={{ marginTop: '16px' }}>
              No Users Registered
            </Title>
            <Text type="secondary">
              Get started by registering your first user in the Registration tab.
            </Text>
          </div>
        </Card>
      )}
    </div>
  );
};

export default UserManagement;