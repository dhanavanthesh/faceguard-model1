import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { 
  Card, 
  Button, 
  Space, 
  Switch, 
  Slider, 
  Alert, 
  Statistic, 
  Row, 
  Col,
  List,
  Badge,
  Typography,
  Divider,
  Select,
  notification,
  Modal,
  Avatar,
  Tag
} from 'antd';
import { 
  VideoCameraOutlined, 
  PauseOutlined, 
  PlayCircleOutlined,
  EyeOutlined,
  UserOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  CameraOutlined,
  CheckCircleOutlined,
  SafetyCertificateOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

import apiService from '../../services/api';

const { Title, Text } = Typography;
const { Option } = Select;

const LiveRecognition = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  // State management
  const [isActive, setIsActive] = useState(false);
  const [recognitionResults, setRecognitionResults] = useState([]);
  const [currentResults, setCurrentResults] = useState([]);
  const [stats, setStats] = useState({
    totalFrames: 0,
    recognizedFaces: 0,
    fps: 0,
    avgConfidence: 0
  });
  
  // Settings
  const [settings, setSettings] = useState({
    interval: 2000, // milliseconds - slower for better recognition
    confidenceThreshold: 0.6,
    maxResults: 50,
    showBoundingBoxes: true,
    videoConstraints: {
      width: 640,
      height: 480,
      facingMode: "user"
    }
  });

  const [error, setError] = useState(null);
  const [processing, setProcessing] = useState(false);
  
  // Recognition popup state
  const [recognitionPopup, setRecognitionPopup] = useState({
    visible: false,
    person: null,
    capturedImage: null,
    timestamp: null
  });
  const [recentRecognitions, setRecentRecognitions] = useState(new Set());

  // Video constraints options
  const videoConstraintsOptions = [
    { label: '640x480', value: { width: 640, height: 480 } },
    { label: '800x600', value: { width: 800, height: 600 } },
    { label: '1280x720', value: { width: 1280, height: 720 } }
  ];

  // Start/Stop live recognition
  const toggleRecognition = useCallback(() => {
    if (isActive) {
      stopRecognition();
    } else {
      startRecognition();
    }
  }, [isActive]);

  const startRecognition = useCallback(() => {
    if (!webcamRef.current) {
      notification.error({
        message: 'Camera Error',
        description: 'Please ensure camera is available and permissions are granted'
      });
      return;
    }

    setIsActive(true);
    setError(null);
    
    intervalRef.current = setInterval(() => {
      captureAndRecognize();
    }, settings.interval);

    notification.success({
      message: 'Live Recognition Started',
      description: 'Real-time face recognition is now active'
    });
  }, [settings.interval]);

  const stopRecognition = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsActive(false);
    setProcessing(false);
    
    notification.info({
      message: 'Live Recognition Stopped',
      description: 'Real-time face recognition has been paused'
    });
  }, []);

  const captureAndRecognize = useCallback(async () => {
    if (!webcamRef.current || processing) {
      return;
    }

    try {
      setProcessing(true);
      
      // Capture image from webcam
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        return;
      }

      // Convert base64 to file for API
      const response = await fetch(imageSrc);
      const blob = await response.blob();
      const file = new File([blob], `live_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });

      // Send to API for recognition  
      const result = await apiService.recognizeFacesFile(file);
      
      // Update current results (adjust for API response format)
      const faces = result.faces || [];
      const processedFaces = faces.map(face => ({
        ...face,
        recognized: face.name !== 'UNKNOWN',
        user_name: face.name
      }));
      
      setCurrentResults(processedFaces);
      
      // Draw face detection results on canvas overlay
      drawFaceResults(faces);
      
      // Show popup for recognized registered users
      console.log('🔍 Face detection results:', faces);
      const registeredFaces = faces.filter(face => {
        console.log('📊 Face check:', { 
          name: face.name, 
          is_registered: face.is_registered, 
          confidence: face.confidence 
        });
        return face.is_registered && face.confidence > 0.4; // Lower threshold for testing
      });
      
      console.log('✅ Registered faces found:', registeredFaces.length);
      
      if (registeredFaces.length > 0) {
        console.log('🚀 Showing popup for:', registeredFaces[0].name);
        // Show popup for the first registered face found
        showRecognitionPopup(registeredFaces[0], imageSrc);
        
        // Also show a simple notification as fallback
        notification.success({
          message: 'Person Recognized!',
          description: `Welcome ${registeredFaces[0].name}! Access Authorized.`,
          placement: 'topRight',
          duration: 2,
          icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />
        });
      }
      
      // Update statistics
      setStats(prev => ({
        totalFrames: prev.totalFrames + 1,
        recognizedFaces: prev.recognizedFaces + processedFaces.filter(f => f.recognized).length,
        fps: 1000 / settings.interval, // Calculate based on interval
        avgConfidence: processedFaces.length > 0 
          ? processedFaces.reduce((sum, face) => sum + face.confidence, 0) / processedFaces.length
          : prev.avgConfidence
      }));

      // Add to results history (keep only recent results)
      if (processedFaces.length > 0) {
        const timestamp = new Date().toLocaleTimeString();
        const newResults = processedFaces.map((face, index) => ({
          ...face,
          timestamp,
          frame_id: `${Date.now()}-${index}`
        }));
        
        setRecognitionResults(prev => 
          [...newResults, ...prev].slice(0, settings.maxResults)
        );
      }

    } catch (error) {
      console.error('Recognition error:', error);
      if (error.message && !error.message.includes('timeout')) {
        setError(error.message);
        stopRecognition();
      }
    } finally {
      setProcessing(false);
    }
  }, [processing, settings.maxResults]);

  // Clear canvas overlay
  const clearCanvas = useCallback(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  // Show recognition popup for registered users
  const showRecognitionPopup = useCallback((person, capturedImage) => {
    console.log('🎉 showRecognitionPopup called for:', person.name);
    const now = Date.now();
    const recentKey = `${person.name}-${Math.floor(now / 10000)}`; // Group by 10 second intervals
    
    // Prevent spam - only show popup once per person per 10 seconds
    if (recentRecognitions.has(recentKey)) {
      console.log('⏳ Popup blocked - too recent for:', person.name);
      return;
    }
    
    console.log('✅ Showing popup for:', person.name);

    setRecentRecognitions(prev => {
      const newSet = new Set(prev);
      newSet.add(recentKey);
      
      // Clean up old entries (older than 1 minute)
      const cutoffTime = Math.floor((now - 60000) / 10000);
      for (const key of newSet) {
        const keyTime = parseInt(key.split('-').pop());
        if (keyTime < cutoffTime) {
          newSet.delete(key);
        }
      }
      
      return newSet;
    });

    setRecognitionPopup({
      visible: true,
      person,
      capturedImage,
      timestamp: new Date().toLocaleTimeString()
    });

    // Auto close after 3 seconds
    setTimeout(() => {
      setRecognitionPopup(prev => ({ ...prev, visible: false }));
    }, 3000);
  }, [recentRecognitions]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Update canvas size when video constraints change
  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = settings.videoConstraints.width;
      canvas.height = settings.videoConstraints.height;
      clearCanvas();
    }
  }, [settings.videoConstraints, clearCanvas]);

  // Clear results
  const clearResults = () => {
    setRecognitionResults([]);
    setCurrentResults([]);
    setStats({
      totalFrames: 0,
      recognizedFaces: 0,
      fps: 0,
      avgConfidence: 0
    });
    clearCanvas();
  };

  // Draw face detection results on canvas overlay
  const drawFaceResults = useCallback((faces) => {
    if (!canvasRef.current || !faces || faces.length === 0) {
      clearCanvas();
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      const isRegistered = face.is_registered;
      const color = isRegistered ? '#52c41a' : '#ff4d4f'; // Green for registered, red for unknown

      // Draw bounding box
      if (face.bbox && face.bbox.length === 4) {
        const [x1, y1, x2, y2] = face.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        const label = isRegistered ? `${face.name} (${(face.confidence * 100).toFixed(1)}%)` : 'UNKNOWN';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 25, textWidth + 10, 20);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(label, x1 + 5, y1 - 8);
      }

      // Draw landmarks (5 key points: left eye, right eye, nose, left mouth, right mouth)
      if (face.landmarks && face.landmarks.length === 5) {
        ctx.fillStyle = color;
        
        face.landmarks.forEach((landmark, index) => {
          const [x, y] = landmark;
          
          // Draw different shapes for different landmarks
          ctx.beginPath();
          
          if (index === 0 || index === 1) {
            // Eyes - small circles
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
          } else if (index === 2) {
            // Nose - triangle
            ctx.moveTo(x, y - 3);
            ctx.lineTo(x - 3, y + 3);
            ctx.lineTo(x + 3, y + 3);
            ctx.closePath();
          } else {
            // Mouth corners - small circles
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
          }
          
          ctx.fill();
        });
      }
    });
  }, [clearCanvas]);

  // Settings change handlers
  const handleIntervalChange = (value) => {
    setSettings(prev => ({ ...prev, interval: value }));
    if (isActive) {
      stopRecognition();
      setTimeout(() => startRecognition(), 100);
    }
  };

  const handleResolutionChange = (value) => {
    setSettings(prev => ({ 
      ...prev, 
      videoConstraints: { 
        ...prev.videoConstraints, 
        ...value 
      } 
    }));
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            📹 Live Recognition
          </Title>
          <Text type="secondary">Real-time face recognition from webcam</Text>
        </Col>
      </Row>

      {error && (
        <Alert
          message="Recognition Error"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Webcam and Controls */}
        <Col xs={24} lg={16}>
          <Card 
            title="Live Camera Feed"
            extra={
              <Space>
                <Badge status={isActive ? 'processing' : 'default'} />
                <Text>{isActive ? 'Active' : 'Stopped'}</Text>
              </Space>
            }
          >
            {/* Webcam */}
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              <div style={{ 
                position: 'relative', 
                display: 'inline-block',
                border: '2px solid #f0f0f0',
                borderRadius: '8px',
                overflow: 'hidden'
              }}>
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={settings.videoConstraints}
                  style={{ display: 'block' }}
                />
                
                {/* Canvas overlay for face landmarks and bounding boxes */}
                <canvas
                  ref={canvasRef}
                  width={settings.videoConstraints.width}
                  height={settings.videoConstraints.height}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    pointerEvents: 'none',
                    zIndex: 1
                  }}
                />
                
                {/* Processing indicator */}
                {processing && (
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'rgba(0,0,0,0.7)',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}>
                    Processing...
                  </div>
                )}

                {/* Face detection overlay with status */}
                {currentResults.length > 0 && (
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    background: 'rgba(0,0,0,0.8)',
                    color: 'white',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: 'bold'
                  }}>
                    <div>{currentResults.length} face(s) detected</div>
                    {currentResults.map((face, index) => (
                      <div key={index} style={{ 
                        marginTop: '4px',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        backgroundColor: face.recognized ? '#52c41a' : '#ff4d4f',
                        fontSize: '11px'
                      }}>
                        {face.recognized ? 
                          `✅ RECOGNIZED: ${face.user_name}` : 
                          '❌ UNKNOWN PERSON'
                        }
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Controls */}
            <Space size="middle" style={{ width: '100%', justifyContent: 'center' }}>
              <Button
                type="primary"
                size="large"
                icon={isActive ? <PauseOutlined /> : <PlayCircleOutlined />}
                onClick={toggleRecognition}
                loading={processing && !isActive}
              >
                {isActive ? 'Stop Recognition' : 'Start Recognition'}
              </Button>
              
              <Button
                icon={<CameraOutlined />}
                onClick={() => {
                  if (webcamRef.current) {
                    captureAndRecognize();
                  }
                }}
                disabled={!webcamRef.current || processing}
              >
                Capture Frame
              </Button>
              
              <Button onClick={clearResults}>
                Clear Results
              </Button>
            </Space>

            {/* Current Detection Results */}
            {currentResults.length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <Text strong>Current Frame Results:</Text>
                <List
                  size="small"
                  dataSource={currentResults}
                  renderItem={(face, index) => (
                    <List.Item>
                      <Space>
                        <Badge 
                          status={face.recognized ? 'success' : 'error'} 
                          text={
                            face.recognized 
                              ? `${face.user_name} (${face.confidence.toFixed(3)})` 
                              : 'Unknown Person'
                          }
                        />
                      </Space>
                    </List.Item>
                  )}
                />
              </div>
            )}
          </Card>
        </Col>

        {/* Settings and Stats */}
        <Col xs={24} lg={8}>
          {/* Statistics */}
          <Card title="Recognition Statistics" style={{ marginBottom: '16px' }}>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title="Frames Processed"
                  value={stats.totalFrames}
                  prefix={<EyeOutlined />}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Faces Recognized"
                  value={stats.recognizedFaces}
                  prefix={<UserOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Current FPS"
                  value={stats.fps.toFixed(1)}
                  prefix={<ThunderboltOutlined />}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Avg Accuracy"
                  value={stats.avgConfidence.toFixed(3)}
                  prefix={<EyeOutlined />}
                />
              </Col>
            </Row>
          </Card>

          {/* Settings */}
          <Card title="Settings" extra={<SettingOutlined />}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div>
                <Text strong>Processing Interval</Text>
                <Slider
                  min={500}
                  max={5000}
                  step={100}
                  value={settings.interval}
                  onChange={handleIntervalChange}
                  marks={{
                    500: '0.5s',
                    1000: '1s',
                    2000: '2s',
                    5000: '5s'
                  }}
                  disabled={isActive}
                />
                <Text type="secondary">Current: {settings.interval}ms</Text>
              </div>

              <div>
                <Text strong>Confidence Threshold</Text>
                <Slider
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  value={settings.confidenceThreshold}
                  onChange={(value) => setSettings(prev => ({ ...prev, confidenceThreshold: value }))}
                  marks={{
                    0.1: '0.1',
                    0.5: '0.5',
                    1.0: '1.0'
                  }}
                />
              </div>

              <div>
                <Text strong>Video Resolution</Text>
                <Select
                  style={{ width: '100%', marginTop: '8px' }}
                  value={JSON.stringify(settings.videoConstraints)}
                  onChange={(value) => handleResolutionChange(JSON.parse(value))}
                  disabled={isActive}
                >
                  {videoConstraintsOptions.map((option, index) => (
                    <Option key={index} value={JSON.stringify(option.value)}>
                      {option.label}
                    </Option>
                  ))}
                </Select>
              </div>

              <div>
                <Text strong>Max Results History</Text>
                <Slider
                  min={10}
                  max={100}
                  step={10}
                  value={settings.maxResults}
                  onChange={(value) => setSettings(prev => ({ ...prev, maxResults: value }))}
                  marks={{
                    10: '10',
                    50: '50',
                    100: '100'
                  }}
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Results History */}
      <Row style={{ marginTop: '16px' }}>
        <Col span={24}>
          <Card 
            title="Recognition History" 
            extra={
              <Text type="secondary">
                Showing last {Math.min(recognitionResults.length, settings.maxResults)} results
              </Text>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={recognitionResults}
              pagination={{
                pageSize: 10,
                size: 'small'
              }}
              renderItem={(result) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Badge status={result.recognized ? 'success' : 'error'}>
                        <UserOutlined style={{ fontSize: '16px' }} />
                      </Badge>
                    }
                    title={
                      result.recognized ? result.user_name : 'Unknown Person'
                    }
                    description={
                      <Space split={<Divider type="vertical" />}>
                        <Text type="secondary">{result.timestamp}</Text>
                        <Text type="secondary">
                          Confidence: {result.confidence.toFixed(3)}
                        </Text>
                        <Text type="secondary">
                          Frame: {result.frame_id}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
              locale={{
                emptyText: isActive ? 'Waiting for recognition results...' : 'No results yet. Start recognition to see results.'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Recognition Success Popup */}
      <Modal
        title={null}
        open={recognitionPopup.visible}
        footer={null}
        closable={false}
        width={400}
        centered
        styles={{
          body: {
            padding: '20px',
            textAlign: 'center',
            background: 'linear-gradient(135deg, #52c41a, #389e0d)',
            color: 'white',
            borderRadius: '8px'
          }
        }}
        style={{ 
          '.ant-modal-content': { 
            background: 'transparent',
            boxShadow: '0 8px 32px rgba(82, 196, 26, 0.3)'
          }
        }}
      >
        {recognitionPopup.person && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <CheckCircleOutlined 
                style={{ 
                  fontSize: '48px', 
                  color: 'white',
                  marginBottom: '16px',
                  display: 'block'
                }} 
              />
              <Title level={3} style={{ color: 'white', margin: 0 }}>
                Access Authorized
              </Title>
            </div>

            {recognitionPopup.capturedImage && (
              <div style={{ 
                width: '120px', 
                height: '120px', 
                margin: '0 auto',
                border: '4px solid white',
                borderRadius: '50%',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <img 
                  src={recognitionPopup.capturedImage}
                  alt="Captured face"
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover'
                  }}
                />
              </div>
            )}

            <div>
              <Title level={4} style={{ color: 'white', margin: '8px 0' }}>
                Welcome, {recognitionPopup.person.name}
              </Title>
              
              <div style={{ marginBottom: '12px' }}>
                <Tag 
                  icon={<SafetyCertificateOutlined />} 
                  color="success" 
                  style={{ 
                    fontSize: '14px',
                    padding: '4px 12px',
                    borderRadius: '16px'
                  }}
                >
                  Registered User
                </Tag>
              </div>

              <Text style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '13px' }}>
                Accuracy: {(recognitionPopup.person.confidence * 100).toFixed(1)}%
              </Text>
              <br />
              <Text style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '13px' }}>
                Time: {recognitionPopup.timestamp}
              </Text>
            </div>
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default LiveRecognition;