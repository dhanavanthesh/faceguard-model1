import React from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Tag,
  Divider,
  Alert,
  Space,
  Badge
} from 'antd';
import {
  InteractionOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  SafetyOutlined,
  HandOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const AvailableSigns = () => {
  // Hand gestures from sign_detection.py
  const handGestures = [
    {
      name: 'HELP',
      displayName: 'Help Signal',
      isEmergency: true,
      description: 'Raise hand palm outward - universal help signal',
      instructions: 'Extend arm upward with open palm facing forward'
    },
    {
      name: 'STOP',
      displayName: 'Stop Signal',
      isEmergency: true,
      description: 'Open palm extended forward - stop/halt gesture',
      instructions: 'Extend arm forward with open palm facing camera'
    },
    {
      name: 'CALL',
      displayName: 'Call Gesture',
      isEmergency: true,
      description: 'Phone call gesture - pinky and thumb extended',
      instructions: 'Make phone shape with thumb and pinky extended'
    },
    {
      name: 'OK',
      displayName: 'OK Sign',
      isEmergency: false,
      description: 'Thumb and index finger circle - everything is okay',
      instructions: 'Form circle with thumb and index finger'
    },
    {
      name: 'THUMBS_UP',
      displayName: 'Thumbs Up',
      isEmergency: false,
      description: 'Positive gesture - approval or agreement',
      instructions: 'Extend thumb upward with closed fist'
    },
    {
      name: 'THUMBS_DOWN',
      displayName: 'Thumbs Down',
      isEmergency: false,
      description: 'Negative gesture - disapproval or disagreement',
      instructions: 'Point thumb downward with closed fist'
    },
    {
      name: 'PEACE',
      displayName: 'Peace Sign',
      isEmergency: false,
      description: 'V-shape with index and middle finger - peace',
      instructions: 'Extend index and middle finger in V-shape'
    },
    {
      name: 'FIST',
      displayName: 'Fist',
      isEmergency: false,
      description: 'Closed fist gesture',
      instructions: 'Close all fingers into a fist'
    },
    {
      name: 'POINT_LEFT',
      displayName: 'Point Left',
      isEmergency: false,
      description: 'Pointing gesture to the left',
      instructions: 'Extend index finger pointing to the left'
    },
    {
      name: 'POINT_RIGHT',
      displayName: 'Point Right',
      isEmergency: false,
      description: 'Pointing gesture to the right',
      instructions: 'Extend index finger pointing to the right'
    },
    {
      name: 'POINT_UP',
      displayName: 'Point Up',
      isEmergency: false,
      description: 'Pointing gesture upward',
      instructions: 'Extend index finger pointing upward'
    },
    {
      name: 'POINT_DOWN',
      displayName: 'Point Down',
      isEmergency: false,
      description: 'Pointing gesture downward',
      instructions: 'Extend index finger pointing downward'
    },
    {
      name: 'COVER_MOUTH',
      displayName: 'Cover Mouth',
      isEmergency: true,
      description: 'Hand covering mouth - silence/distress signal',
      instructions: 'Place hand over mouth area'
    },
    {
      name: 'TOUCH_EAR',
      displayName: 'Touch Ear',
      isEmergency: true,
      description: 'Hand touching ear - hearing problem/distress',
      instructions: 'Touch or hold hand near ear'
    },
    {
      name: 'WAVE',
      displayName: 'Wave',
      isEmergency: false,
      description: 'Greeting or goodbye wave',
      instructions: 'Wave hand side to side with open palm'
    }
  ];

  // Pose gestures from sign_detection.py
  const poseGestures = [
    {
      name: 'HANDS_UP',
      displayName: 'Hands Up',
      isEmergency: true,
      description: 'Both hands raised above head - surrender/emergency',
      instructions: 'Raise both arms above your head'
    },
    {
      name: 'ARMS_CROSSED',
      displayName: 'Arms Crossed',
      isEmergency: true,
      description: 'Arms crossed over chest - defensive/distress',
      instructions: 'Cross both arms over your chest'
    },
    {
      name: 'CHEST_TAP',
      displayName: 'Chest Tap',
      isEmergency: true,
      description: 'Hand tapping chest - medical emergency',
      instructions: 'Tap chest area with hand'
    },
    {
      name: 'HOLD_NECK',
      displayName: 'Hold Neck',
      isEmergency: true,
      description: 'Hand holding neck - choking/breathing difficulty',
      instructions: 'Place hand on neck/throat area'
    },
    {
      name: 'LYING_DOWN',
      displayName: 'Lying Down',
      isEmergency: true,
      description: 'Person lying down - medical emergency',
      instructions: 'Lie down horizontally'
    },
    {
      name: 'KNEELING',
      displayName: 'Kneeling',
      isEmergency: true,
      description: 'Person kneeling - distress position',
      instructions: 'Kneel on one or both knees'
    },
    {
      name: 'HANDS_ON_HIPS',
      displayName: 'Hands on Hips',
      isEmergency: false,
      description: 'Both hands placed on hips',
      instructions: 'Place both hands on your hips'
    },
    {
      name: 'CROSS_ARMS_ABOVE_HEAD',
      displayName: 'Cross Arms Above Head',
      isEmergency: true,
      description: 'Arms crossed above head - distress signal',
      instructions: 'Cross both arms above your head forming an X'
    }
  ];

  const emergencyGestures = [...handGestures, ...poseGestures].filter(g => g.isEmergency);
  const normalGestures = [...handGestures, ...poseGestures].filter(g => !g.isEmergency);

  const GestureCard = ({ gesture }) => (
    <Card 
      size="small"
      style={{ 
        marginBottom: '16px',
        border: gesture.isEmergency ? '2px solid #ff4d4f' : '1px solid #d9d9d9'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <Space align="start">
            <InteractionOutlined 
              style={{ 
                color: gesture.isEmergency ? '#ff4d4f' : '#1890ff',
                fontSize: '18px'
              }} 
            />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Text strong>{gesture.displayName}</Text>
                <Tag color={gesture.isEmergency ? 'red' : 'blue'} size="small">
                  {gesture.isEmergency ? 'EMERGENCY' : 'NORMAL'}
                </Tag>
              </div>
              <Paragraph style={{ margin: '8px 0 4px 0', fontSize: '14px' }}>
                {gesture.description}
              </Paragraph>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                <strong>How to:</strong> {gesture.instructions}
              </Text>
            </div>
          </Space>
        </div>
      </div>
    </Card>
  );

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: 'calc(100vh - 64px)' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0 }}>
          <InteractionOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          Available Signs & Gestures
        </Title>
        <Text type="secondary">
          Complete list of detectable gestures for communication and emergency situations
        </Text>
      </div>

      {/* Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#1890ff' }}>
                {handGestures.length + poseGestures.length}
              </div>
              <Text type="secondary">Total Gestures</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#ff4d4f' }}>
                {emergencyGestures.length}
              </div>
              <Text type="secondary">Emergency Signs</Text>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#52c41a' }}>
                {normalGestures.length}
              </div>
              <Text type="secondary">Normal Gestures</Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Emergency Alert */}
      <Alert
        message="Emergency Gesture Detection"
        description="Emergency gestures trigger immediate alerts and notifications. These are designed to help in distress situations."
        type="warning"
        icon={<WarningOutlined />}
        showIcon
        style={{ marginBottom: '24px' }}
      />

      <Row gutter={[24, 24]}>
        {/* Emergency Gestures */}
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <Badge status="error" />
                <WarningOutlined style={{ color: '#ff4d4f' }} />
                <span>Emergency Gestures ({emergencyGestures.length})</span>
              </Space>
            }
            style={{ height: '100%' }}
          >
            <Text type="secondary" style={{ marginBottom: '16px', display: 'block' }}>
              These gestures trigger emergency alerts and high-priority notifications
            </Text>
            {emergencyGestures.map((gesture, index) => (
              <GestureCard key={`emergency-${index}`} gesture={gesture} />
            ))}
          </Card>
        </Col>

        {/* Normal Gestures */}
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <Badge status="success" />
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
                <span>Normal Gestures ({normalGestures.length})</span>
              </Space>
            }
            style={{ height: '100%' }}
          >
            <Text type="secondary" style={{ marginBottom: '16px', display: 'block' }}>
              Standard communication and interaction gestures
            </Text>
            {normalGestures.map((gesture, index) => (
              <GestureCard key={`normal-${index}`} gesture={gesture} />
            ))}
          </Card>
        </Col>
      </Row>

      {/* Usage Instructions */}
      <Card title="How to Use Gesture Detection" style={{ marginTop: '24px' }}>
        <Row gutter={[24, 24]}>
          <Col xs={24} md={12}>
            <Title level={4}>📱 For Live Detection:</Title>
            <ol>
              <li>Go to "Sign Detection" page</li>
              <li>Allow camera access</li>
              <li>Click "Start Detection"</li>
              <li>Perform gestures in front of the camera</li>
              <li>System will detect and alert for emergency gestures</li>
            </ol>
          </Col>
          <Col xs={24} md={12}>
            <Title level={4}>💡 Tips for Better Detection:</Title>
            <ul>
              <li>Ensure good lighting conditions</li>
              <li>Keep hands/body clearly visible</li>
              <li>Hold gestures for 2-3 seconds</li>
              <li>Maintain steady position</li>
              <li>Emergency gestures have higher detection priority</li>
            </ul>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default AvailableSigns;