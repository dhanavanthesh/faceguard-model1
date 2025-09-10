import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Alert,
  Badge,
  Spin,
  Typography,
  Space,
  Progress,
  List,
  Statistic,
  Switch,
  Modal,
  notification,
  Divider,
  Tag
} from 'antd';
import {
  VideoCameraOutlined,
  StopOutlined,
  ExclamationCircleOutlined,
  InteractionOutlined,
  EyeOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SoundOutlined,
  MutedOutlined
} from '@ant-design/icons';
import Webcam from 'react-webcam';
import apiService from '../../services/api';

const { Title, Text } = Typography;

const SignDetection = () => {
  // State management
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentGesture, setCurrentGesture] = useState(null);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [webcamError, setWebcamError] = useState(null);
  const [emergencyAlert, setEmergencyAlert] = useState(null);
  const [statistics, setStatistics] = useState(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const [detectionRate, setDetectionRate] = useState(500); // ms - faster for better gesture responsiveness
  const [processing, setProcessing] = useState(false);

  // Refs
  const webcamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const isDetectingRef = useRef(false);

  // Webcam configuration
  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  // Core supported gestures
  const supportedGestures = {
    'THUMBS_UP': {
      name: 'Thumbs Up',
      icon: '👍',
      description: 'Make a fist with thumb pointing up',
      color: '#52c41a'
    },
    'PALM_STOP': {
      name: 'Palm Stop',
      icon: '✋',
      description: 'Show open palm with all fingers extended',
      color: '#faad14'
    },
    'ARMS_CROSSED': {
      name: 'Arms Crossed',
      icon: '❌',
      description: 'Cross both arms over chest (Emergency)',
      color: '#ff4d4f'
    }
  };

  // Initialize component
  useEffect(() => {
    console.log('🔧 SignDetection: Component initializing...');
    console.log('🔧 SignDetection: API Service available:', !!apiService);
    console.log('🔧 SignDetection: API functions:', Object.getOwnPropertyNames(apiService));
    
    checkSystemStatus();
    loadStatistics();
    
    // Set up periodic updates
    const statusInterval = setInterval(checkSystemStatus, 30000);
    const statsInterval = setInterval(loadStatistics, 60000);
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(statsInterval);
      stopDetection();
    };
  }, []);

  // Check sign detection system status
  const checkSystemStatus = async () => {
    console.log('🔧 SignDetection: checkSystemStatus called');
    try {
      console.log('🔧 SignDetection: Calling apiService.getSignDetectionStatus()');
      console.log('🔧 SignDetection: API function exists:', typeof apiService.getSignDetectionStatus);
      
      const response = await apiService.getSignDetectionStatus();
      console.log('🔧 SignDetection: API response received:', response);
      
      setSystemStatus(response);
      setLoading(false);
      
      if (!response.success || response.status !== 'running') {
        console.log('🔧 SignDetection: System not operational, showing warning');
        notification.warning({
          message: 'Sign Detection System',
          description: 'Sign detection system is not fully operational',
          placement: 'topRight'
        });
      } else {
        console.log('🔧 SignDetection: System operational!');
      }
    } catch (error) {
      console.error('🔧 SignDetection: Failed to check system status:', error);
      console.error('🔧 SignDetection: Error details:', error.message, error.stack);
      setSystemStatus({ success: false, status: 'error' });
      setLoading(false);
    }
  };

  // Load detection statistics
  const loadStatistics = async () => {
    console.log('🔧 SignDetection: loadStatistics called');
    try {
      console.log('🔧 SignDetection: Calling apiService.getSignStatistics()');
      console.log('🔧 SignDetection: API function exists:', typeof apiService.getSignStatistics);
      
      const response = await apiService.getSignStatistics();
      console.log('🔧 SignDetection: Statistics response:', response);
      
      if (response.success) {
        setStatistics(response.statistics);
        console.log('🔧 SignDetection: Statistics loaded successfully');
      }
    } catch (error) {
      console.error('🔧 SignDetection: Failed to load statistics:', error);
      console.error('🔧 SignDetection: Statistics error details:', error.message, error.stack);
    }
  };

  // Load detection history
  const loadDetectionHistory = async () => {
    try {
      const response = await apiService.getSignDetectionLogs(20);
      if (response.success) {
        setDetectionHistory(response.logs);
      }
    } catch (error) {
      console.error('Failed to load detection history:', error);
    }
  };

  // Play alert sound for emergencies
  const playAlertSound = useCallback(() => {
    if (!soundEnabled) return;
    
    try {
      // Create audio context if not exists
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      const audioContext = audioContextRef.current;
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
      oscillator.type = 'sine';
      
      gainNode.gain.setValueAtTime(0, audioContext.currentTime);
      gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.1);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
      
    } catch (error) {
      console.warn('Failed to play alert sound:', error);
    }
  }, [soundEnabled]);

  // Update ref when state changes
  useEffect(() => {
    isDetectingRef.current = isDetecting;
  }, [isDetecting]);

  // Capture and process frame
  const captureFrame = useCallback(async () => {
    if (!webcamRef.current || !isDetectingRef.current || processing) {
      return;
    }
    
    try {
      setProcessing(true);
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        setProcessing(false);
        return;
      }
      
      const response = await apiService.detectSigns({
        image_data: imageSrc
      });
      
      if (response.success) {
        const detectionResult = {
          gesture: response.gesture,
          confidence: response.confidence,
          is_emergency: response.is_emergency,
          hands_detected: response.hands_detected,
          landmarks: response.landmarks,
          timestamp: response.timestamp || new Date().toISOString()
        };
        
        // Update current gesture only if confidence is above threshold
        if (response.confidence >= confidenceThreshold && response.gesture !== 'NONE') {
          setCurrentGesture(detectionResult);
          
          // Handle emergency gestures
          if (response.is_emergency && response.confidence >= 0.8) {
            setEmergencyAlert(detectionResult);
            playAlertSound();
            
            notification.error({
              message: 'EMERGENCY GESTURE DETECTED',
              description: `${response.gesture} detected with ${(response.confidence * 100).toFixed(1)}% confidence`,
              placement: 'topRight',
              duration: 8
            });
            
            // Auto-clear emergency alert after 10 seconds
            setTimeout(() => setEmergencyAlert(null), 10000);
          }
          
          // Update detection history
          setDetectionHistory(prev => [detectionResult, ...prev.slice(0, 19)]);
        } else if (response.gesture === 'NONE' || response.confidence < confidenceThreshold) {
          setCurrentGesture(null);
        }
      }
      
    } catch (error) {
      console.error('Detection error:', error);
      
      // Show error notification
      if (error.response?.status === 503) {
        notification.warning({
          message: 'Sign Detection Not Ready',
          description: 'Sign detection system is not initialized or models are not trained',
          placement: 'topRight',
          duration: 3
        });
        stopDetection();
      } else if (error.response?.status === 500) {
        notification.error({
          message: 'Detection Error',
          description: 'Error processing gesture detection',
          placement: 'topRight',
          duration: 3
        });
      }
    } finally {
      setProcessing(false);
    }
  }, [confidenceThreshold, playAlertSound, processing]);

  // Start detection
  const startDetection = () => {
    console.log('Start detection clicked', { webcamRef: !!webcamRef.current, systemStatus });
    
    if (!webcamRef.current) {
      console.error('Webcam not available');
      notification.error({
        message: 'Camera Error',
        description: 'Please ensure camera access is granted',
        placement: 'topRight'
      });
      return;
    }
    
    console.log('Starting sign detection...', { detectionRate, systemStatus });
    setIsDetecting(true);
    isDetectingRef.current = true; // Set ref immediately
    setCurrentGesture(null);
    setEmergencyAlert(null);
    
    // Start detection interval
    detectionIntervalRef.current = setInterval(() => {
      console.log('Interval triggered, calling captureFrame');
      captureFrame();
    }, detectionRate);
    console.log('Detection interval started with rate:', detectionRate, 'Interval ID:', detectionIntervalRef.current);
    
    notification.success({
      message: 'Sign Detection Started',
      description: 'Real-time gesture recognition is now active',
      placement: 'topRight'
    });
  };

  // Stop detection
  const stopDetection = () => {
    setIsDetecting(false);
    isDetectingRef.current = false; // Set ref immediately
    setCurrentGesture(null);
    setProcessing(false);
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    notification.info({
      message: 'Sign Detection Stopped',
      description: 'Gesture recognition has been paused',
      placement: 'topRight'
    });
  };

  // Handle webcam error
  const handleWebcamError = (error) => {
    console.error('Webcam error:', error);
    setWebcamError('Camera access failed. Please check permissions and try again.');
    setIsDetecting(false);
  };

  // Clear emergency alert
  const clearEmergencyAlert = () => {
    setEmergencyAlert(null);
  };

  // Format gesture name for display
  const formatGestureName = (gesture) => {
    if (!gesture || gesture === 'NONE') return 'No gesture';
    const gestureInfo = supportedGestures[gesture];
    return gestureInfo ? gestureInfo.name : gesture.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase());
  };

  // Get gesture info
  const getGestureInfo = (gesture) => {
    return supportedGestures[gesture] || {
      name: 'Unknown',
      icon: '❓',
      description: 'Unknown gesture',
      color: '#d9d9d9'
    };
  };

  // Get gesture color based on type
  const getGestureColor = (gesture, isEmergency) => {
    if (isEmergency) return '#ff4d4f';
    if (!gesture || gesture === 'NONE') return '#d9d9d9';
    const gestureInfo = getGestureInfo(gesture);
    return gestureInfo.color;
  };

  // Render system status indicator
  const renderSystemStatus = () => {
    if (!systemStatus) return null;
    
    const statusColor = systemStatus.success && systemStatus.status === 'running' ? 'success' : 'error';
    const statusText = systemStatus.success && systemStatus.status === 'running' ? 'Operational' : 'Not Available';
    
    return (
      <Badge 
        status={statusColor} 
        text={
          <Text strong={statusColor === 'success'}>
            Sign Detection: {statusText}
          </Text>
        } 
      />
    );
  };

  if (loading) {
    return (
      <div style={{ 
        height: '60vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        flexDirection: 'column'
      }}>
        <Spin size="large" />
        <Text style={{ marginTop: '16px' }}>
          Initializing Sign Detection System...
        </Text>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: 'calc(100vh - 64px)' }}>
      {/* Emergency Alert Modal */}
      <Modal
        title={
          <div style={{ color: '#ff4d4f' }}>
            <ExclamationCircleOutlined style={{ marginRight: '8px' }} />
            🚨 EMERGENCY GESTURE DETECTED
          </div>
        }
        open={!!emergencyAlert}
        onCancel={clearEmergencyAlert}
        footer={[
          <Button key="ok" type="primary" danger onClick={clearEmergencyAlert}>
            Acknowledge
          </Button>
        ]}
        centered
      >
        {emergencyAlert && (
          <div>
            <p><strong>Gesture:</strong> {formatGestureName(emergencyAlert.gesture)}</p>
            <p><strong>Confidence:</strong> {(emergencyAlert.confidence * 100).toFixed(1)}%</p>
            <p><strong>Time:</strong> {new Date(emergencyAlert.timestamp).toLocaleTimeString()}</p>
            <Alert
              message="Emergency Protocol Activated"
              description="This gesture may indicate a person in distress. Please verify the situation immediately."
              type="error"
              showIcon
            />
          </div>
        )}
      </Modal>

      {/* Header */}
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <Title level={2} style={{ margin: 0 }}>
                  <InteractionOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
                  Simplified Sign Detection
                </Title>
                <Text type="secondary">3 core gestures: Thumbs Up, Palm Stop, and Arms Crossed (Emergency)</Text>
              </div>
              <Space>
                {renderSystemStatus()}
                <Switch
                  checked={soundEnabled}
                  onChange={setSoundEnabled}
                  checkedChildren={<SoundOutlined />}
                  unCheckedChildren={<MutedOutlined />}
                  title="Emergency Sound Alerts"
                />
              </Space>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        {/* Camera and Controls */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <VideoCameraOutlined />
                Live Camera Feed
              </Space>
            }
            extra={
              <Space>
                {!isDetecting ? (
                  <Button
                    type="primary"
                    icon={<EyeOutlined />}
                    onClick={startDetection}
                    disabled={!systemStatus?.success || systemStatus?.status !== 'running'}
                    loading={processing && !isDetecting}
                  >
                    Start Detection
                  </Button>
                ) : (
                  <Button
                    type="default"
                    danger
                    icon={<StopOutlined />}
                    onClick={stopDetection}
                  >
                    Stop Detection
                  </Button>
                )}
                {processing && isDetecting && (
                  <div style={{ display: 'flex', alignItems: 'center', marginLeft: '8px' }}>
                    <Spin size="small" />
                    <span style={{ marginLeft: '8px', fontSize: '12px' }}>Processing...</span>
                  </div>
                )}
              </Space>
            }
          >
            <div style={{ textAlign: 'center' }}>
              {webcamError ? (
                <Alert
                  message="Camera Error"
                  description={webcamError}
                  type="error"
                  showIcon
                  action={
                    <Button size="small" onClick={() => setWebcamError(null)}>
                      Retry
                    </Button>
                  }
                />
              ) : (
                <div style={{ position: 'relative', display: 'inline-block' }}>
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    height={360}
                    screenshotFormat="image/jpeg"
                    width={640}
                    videoConstraints={videoConstraints}
                    onUserMediaError={handleWebcamError}
                    style={{ borderRadius: '8px', maxWidth: '100%' }}
                  />
                  
                  {/* Detection overlay */}
                  {isDetecting && (
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      left: '10px',
                      background: 'rgba(0, 0, 0, 0.7)',
                      color: 'white',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '12px'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ 
                          width: '8px', 
                          height: '8px', 
                          borderRadius: '50%', 
                          backgroundColor: '#ff4d4f',
                          animation: 'pulse 1s infinite'
                        }} />
                        LIVE DETECTION
                        {currentGesture && currentGesture.hands_detected > 0 && (
                          <span style={{ marginLeft: '8px', fontSize: '10px' }}>
                            ({currentGesture.hands_detected} hand{currentGesture.hands_detected > 1 ? 's' : ''})
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Hand landmarks overlay - shows dots for detected hands */}
                  {currentGesture && currentGesture.landmarks && currentGesture.landmarks.hands && currentGesture.landmarks.hands.length > 0 && (
                    <svg style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      pointerEvents: 'none'
                    }}>
                      {currentGesture.landmarks.hands.map((hand, handIndex) => 
                        hand.landmarks && hand.landmarks.landmark ? (
                          <g key={handIndex}>
                            {/* Draw hand landmarks as small circles */}
                            {hand.landmarks.landmark.slice(0, 21).map((landmark, i) => (
                              <circle
                                key={`${handIndex}-${i}`}
                                cx={landmark.x * 640}
                                cy={landmark.y * 360}
                                r="3"
                                fill={hand.handedness === 'Left' ? '#00ff00' : '#ff0000'}
                                opacity="0.8"
                              />
                            ))}
                            
                            {/* Draw hand connections */}
                            {[
                              // Thumb
                              [0, 1], [1, 2], [2, 3], [3, 4],
                              // Index finger
                              [0, 5], [5, 6], [6, 7], [7, 8],
                              // Middle finger
                              [0, 9], [9, 10], [10, 11], [11, 12],
                              // Ring finger
                              [0, 13], [13, 14], [14, 15], [15, 16],
                              // Pinky
                              [0, 17], [17, 18], [18, 19], [19, 20]
                            ].map(([start, end], connectionIndex) => {
                              if (hand.landmarks.landmark[start] && hand.landmarks.landmark[end]) {
                                const startPoint = hand.landmarks.landmark[start];
                                const endPoint = hand.landmarks.landmark[end];
                                return (
                                  <line
                                    key={`${handIndex}-line-${connectionIndex}`}
                                    x1={startPoint.x * 640}
                                    y1={startPoint.y * 360}
                                    x2={endPoint.x * 640}
                                    y2={endPoint.y * 360}
                                    stroke={hand.handedness === 'Left' ? '#00ff00' : '#ff0000'}
                                    strokeWidth="2"
                                    opacity="0.6"
                                  />
                                );
                              }
                              return null;
                            })}
                          </g>
                        ) : null
                      )}
                    </svg>
                  )}
                  
                  {/* Current gesture overlay */}
                  {currentGesture && currentGesture.gesture !== 'NONE' && (
                    <div style={{
                      position: 'absolute',
                      bottom: '10px',
                      left: '10px',
                      right: '10px',
                      background: 'rgba(0, 0, 0, 0.8)',
                      color: 'white',
                      padding: '12px',
                      borderRadius: '6px',
                      textAlign: 'center'
                    }}>
                      <div style={{ 
                        fontSize: '18px', 
                        fontWeight: 'bold',
                        color: getGestureColor(currentGesture.gesture, currentGesture.is_emergency)
                      }}>
                        {getGestureInfo(currentGesture.gesture).icon} {formatGestureName(currentGesture.gesture)}
                        {currentGesture.is_emergency && (
                          <Tag color="red" style={{ marginLeft: '8px' }}>EMERGENCY</Tag>
                        )}
                      </div>
                      <div style={{ fontSize: '14px', marginTop: '4px' }}>
                        Confidence: {(currentGesture.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* Detection Settings */}
            <Divider />
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>Confidence Threshold:</Text>
                <br />
                <Text type="secondary">{(confidenceThreshold * 100).toFixed(0)}%</Text>
                <div style={{ marginTop: '8px' }}>
                  <input
                    type="range"
                    min="0.3"
                    max="0.9"
                    step="0.1"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </Col>
              <Col span={12}>
                <Text strong>Detection Rate:</Text>
                <br />
                <Text type="secondary">{detectionRate}ms</Text>
                <div style={{ marginTop: '8px' }}>
                  <input
                    type="range"
                    min="500"
                    max="2000"
                    step="250"
                    value={detectionRate}
                    onChange={(e) => setDetectionRate(parseInt(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Current Detection and Stats */}
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            {/* Current Detection */}
            <Card
              title={
                <Space>
                  <InteractionOutlined />
                  Current Detection
                </Space>
              }
            >
              {currentGesture && currentGesture.gesture !== 'NONE' ? (
                <div>
                  <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                    <Title level={3} style={{ 
                      margin: 0,
                      color: getGestureColor(currentGesture.gesture, currentGesture.is_emergency)
                    }}>
                      {formatGestureName(currentGesture.gesture)}
                    </Title>
                    {currentGesture.is_emergency && (
                      <Tag color="red" style={{ marginTop: '8px' }}>
                        <WarningOutlined /> EMERGENCY GESTURE
                      </Tag>
                    )}
                  </div>
                  
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="Confidence"
                        value={currentGesture.confidence * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{ 
                          color: currentGesture.confidence > 0.8 ? '#3f8600' : '#faad14' 
                        }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Hands"
                        value={currentGesture.hands_detected}
                        suffix="detected"
                        prefix={<InteractionOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="Emergency"
                        value={currentGesture.is_emergency ? 'Yes' : 'No'}
                        valueStyle={{ 
                          color: currentGesture.is_emergency ? '#ff4d4f' : '#52c41a' 
                        }}
                        prefix={currentGesture.is_emergency ? <WarningOutlined /> : <CheckCircleOutlined />}
                      />
                    </Col>
                  </Row>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Title level={4} type="secondary">
                    {isDetecting ? 'No gesture detected' : 'Detection not active'}
                  </Title>
                  <Text type="secondary">
                    {isDetecting 
                      ? 'Make a gesture in front of the camera' 
                      : 'Click "Start Detection" to begin'
                    }
                  </Text>
                </div>
              )}
            </Card>

            {/* Gesture Guide */}
            <Card title="Gesture Guide" extra={<Text type="secondary">How to perform gestures</Text>}>
              <Space direction="vertical" style={{ width: '100%' }}>
                {Object.entries(supportedGestures).map(([key, gesture]) => (
                  <Card size="small" key={key} style={{ 
                    border: `2px solid ${gesture.color}20`,
                    borderLeft: `4px solid ${gesture.color}`
                  }}>
                    <Row align="middle">
                      <Col span={4}>
                        <div style={{ 
                          fontSize: '24px', 
                          textAlign: 'center',
                          color: gesture.color
                        }}>
                          {gesture.icon}
                        </div>
                      </Col>
                      <Col span={20}>
                        <div>
                          <Text strong style={{ color: gesture.color }}>
                            {gesture.name}
                            {key === 'ARMS_CROSSED' && (
                              <Tag color="red" size="small" style={{ marginLeft: '8px' }}>
                                EMERGENCY
                              </Tag>
                            )}
                          </Text>
                        </div>
                        <div style={{ marginTop: '4px' }}>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {gesture.description}
                          </Text>
                        </div>
                      </Col>
                    </Row>
                  </Card>
                ))}
              </Space>
            </Card>

            {/* Statistics */}
            {statistics && (
              <Card title="Detection Statistics">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="Total Detections"
                      value={statistics.total_detections || 0}
                      prefix={<EyeOutlined />}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="Emergency Rate"
                      value={statistics.emergency_rate || 0}
                      precision={1}
                      suffix="%"
                      valueStyle={{ 
                        color: (statistics.emergency_rate || 0) > 10 ? '#ff4d4f' : '#52c41a' 
                      }}
                      prefix={<WarningOutlined />}
                    />
                  </Col>
                </Row>
                <Divider />
                <div>
                  <Text strong>Most Common Gesture: </Text>
                  <Text>{formatGestureName(statistics.most_common_gesture) || 'None'}</Text>
                </div>
                <div style={{ marginTop: '8px' }}>
                  <Text strong>Average Confidence: </Text>
                  <Text>{((statistics.avg_confidence || 0) * 100).toFixed(1)}%</Text>
                </div>
              </Card>
            )}
          </Space>
        </Col>
      </Row>

      {/* Detection History */}
      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card
            title="Detection History"
            extra={
              <Button onClick={loadDetectionHistory}>
                Refresh
              </Button>
            }
          >
            <List
              dataSource={detectionHistory}
              locale={{ emptyText: 'No detections recorded yet' }}
              renderItem={(item, index) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Badge 
                        status={item.is_emergency ? "error" : "success"}
                        icon={item.is_emergency ? <WarningOutlined /> : <CheckCircleOutlined />}
                      />
                    }
                    title={
                      <Space>
                        <span>{formatGestureName(item.gesture)}</span>
                        {item.is_emergency && (
                          <Tag color="red" size="small">EMERGENCY</Tag>
                        )}
                        <Text type="secondary">
                          {(item.confidence * 100).toFixed(1)}% confidence
                        </Text>
                      </Space>
                    }
                    description={
                      <Space split={<span style={{ color: '#d9d9d9' }}>|</span>}>
                        <Text type="secondary">
                          {new Date(item.timestamp).toLocaleString()}
                        </Text>
                        <Text type="secondary">
                          {item.hands_detected} hand{item.hands_detected !== 1 ? 's' : ''} detected
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
              pagination={{
                pageSize: 10,
                size: 'small',
                showSizeChanger: false
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* CSS for pulse animation */}
      <style jsx="true">{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default SignDetection;