import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { 
  Card, 
  Button, 
  Form, 
  Input, 
  Space, 
  Alert, 
  Modal,
  Steps,
  Upload,
  Row,
  Col,
  Typography,
  Divider,
  notification,
  Image,
  Spin
} from 'antd';
import { 
  UserAddOutlined,
  CameraOutlined,
  UploadOutlined,
  CheckCircleOutlined,
  EditOutlined,
  ReloadOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

import apiService from '../../services/api';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;

const Registration = () => {
  const webcamRef = useRef(null);
  const [form] = Form.useForm();

  // State management
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [registrationResult, setRegistrationResult] = useState(null);
  const [error, setError] = useState(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const [registrationMethod, setRegistrationMethod] = useState('webcam'); // 'webcam' or 'upload'
  const [webcamReady, setWebcamReady] = useState(false);
  const [webcamError, setWebcamError] = useState(null);
  const [isSecureContext, setIsSecureContext] = useState(window.isSecureContext || window.location.protocol === 'https:');
  const [webcamInitializing, setWebcamInitializing] = useState(false);

  // Enhanced logging helper
  const log = (message, data = null) => {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    console.log(`🎥 [Registration ${timestamp}] ${message}`, data ? data : '');
  };

  // Log initial state
  useEffect(() => {
    log('Component mounted');
    log('Initial security context', {
      isSecureContext: window.isSecureContext,
      protocol: window.location.protocol,
      hostname: window.location.hostname
    });
    log('Browser support check', {
      hasGetUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      hasWebcam: !!window.HTMLMediaElement,
      userAgent: navigator.userAgent
    });

    // Expose inspection helper to global scope
    window.inspectWebcam = () => {
      console.log('🔍 WEBCAM INSPECTION REPORT:', {
        currentStep,
        registrationMethod,
        showWebcam,
        webcamReady,
        webcamInitializing,
        hasWebcamRef: !!webcamRef.current,
        hasError: !!error,
        hasWebcamError: !!webcamError,
        hasCapturedImage: !!capturedImage,
        isSecureContext,
        videoConstraints
      });
    };
    log('Added window.inspectWebcam() helper function');
  }, []);

  // Log state changes
  useEffect(() => {
    log('State change', {
      currentStep,
      registrationMethod,
      showWebcam,
      webcamReady,
      webcamInitializing,
      hasError: !!error,
      hasCapturedImage: !!capturedImage
    });
  }, [currentStep, registrationMethod, showWebcam, webcamReady, webcamInitializing, error, capturedImage]);

  // Webcam settings - simplified for better compatibility
  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
    frameRate: { ideal: 30, max: 30 }
  };

  // Step 0: Choose registration method and enter name
  const handleStepOne = (values) => {
    log('handleStepOne called', { values, registrationMethod });
    setError(null);
    setWebcamError(null);
    
    if (registrationMethod === 'webcam') {
      log('Proceeding to webcam step');
      setCurrentStep(1);
      setShowWebcam(true);
      setWebcamInitializing(true);
      log('State set', { currentStep: 1, showWebcam: true, webcamInitializing: true });
    } else {
      log('Proceeding to upload step');
      setCurrentStep(1); // Both go to step 1, but different rendering
      setShowWebcam(false);
      setWebcamInitializing(false);
      log('State set', { currentStep: 1, showWebcam: false, webcamInitializing: false });
    }
  };

  // Webcam event handlers
  const handleWebcamUserMedia = useCallback((stream) => {
    log('✅ Webcam access granted', {
      streamId: stream.id,
      streamActive: stream.active,
      videoTracks: stream.getVideoTracks().length,
      audioTracks: stream.getAudioTracks().length,
      videoTrackSettings: stream.getVideoTracks()[0]?.getSettings()
    });
    setWebcamReady(true);
    setWebcamInitializing(false);
    setWebcamError(null);
    setError(null);
    
    notification.success({
      message: 'Camera Ready',
      description: 'Webcam is now active and ready for capture',
      placement: 'topRight'
    });
  }, []);

  const handleWebcamUserMediaError = useCallback((error) => {
    log('❌ Webcam access error', {
      error: error.message,
      name: error.name,
      constraintName: error.constraintName,
      stack: error.stack
    });
    setWebcamReady(false);
    setWebcamInitializing(false);
    setWebcamError(error.message || 'Webcam access denied');
    setError(`Webcam access denied: ${error.message || 'Please allow camera access and refresh the page'}`);
    
    notification.error({
      message: 'Camera Access Denied',
      description: error.message || 'Please allow camera access when prompted by your browser',
      placement: 'topRight'
    });
  }, []);

  // Step 1: Capture photo from webcam
  const capturePhoto = useCallback(() => {
    log('capturePhoto called');
    log('Webcam ref status', {
      refExists: !!webcamRef.current,
      refReady: webcamReady,
      webcamInitializing,
      currentStep
    });

    if (!webcamRef.current) {
      log('ERROR: Webcam ref not available');
      setError('Webcam not available. Please ensure camera access is granted.');
      notification.error({
        message: 'Camera Error',
        description: 'Webcam reference not available'
      });
      return;
    }

    if (!webcamReady) {
      log('ERROR: Webcam not ready');
      setError('Webcam is not ready yet. Please wait for camera to initialize.');
      notification.error({
        message: 'Camera Not Ready',
        description: 'Please wait for camera to initialize'
      });
      return;
    }

    try {
      log('Attempting to capture screenshot');
      const imageSrc = webcamRef.current.getScreenshot();
      log('Screenshot result', {
        hasImage: !!imageSrc,
        imageLength: imageSrc ? imageSrc.length : 0,
        imageStart: imageSrc ? imageSrc.substring(0, 50) : 'null'
      });
      
      if (imageSrc) {
        setCapturedImage(imageSrc);
        setShowWebcam(false);
        setCurrentStep(2);
        log('Photo captured successfully, moving to step 2');
        
        notification.success({
          message: 'Photo Captured',
          description: 'Photo captured successfully! Please review and submit.'
        });
      } else {
        throw new Error('Failed to capture image - screenshot returned null');
      }
    } catch (error) {
      log('ERROR: Capture failed', error);
      setError('Could not capture image from webcam. Please try again.');
      notification.error({
        message: 'Capture Failed',
        description: 'Could not capture image from webcam'
      });
    }
  }, [webcamReady, webcamInitializing, currentStep]);

  // Handle file upload
  const handleFileUpload = (info) => {
    const { file } = info;
    
    if (file.status === 'uploading') {
      return;
    }

    // Validate file
    const validation = apiService.validateImageFile(file.originFileObj || file);
    if (!validation.valid) {
      notification.error({
        message: 'Invalid File',
        description: validation.message
      });
      return;
    }

    // Convert to base64 for preview
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedFile(reader.result);
      setCapturedImage(reader.result);
      setCurrentStep(2);
    };
    reader.readAsDataURL(file.originFileObj || file);
  };

  // Step 2: Review and submit registration
  const handleRegistration = async (values) => {
    try {
      setLoading(true);
      setError(null);

      const { name } = values;
      const imageData = capturedImage;

      if (!name || !imageData) {
        throw new Error('Name and image are required');
      }

      // Convert base64 to file for API
      const response = await fetch(imageData);
      const blob = await response.blob();
      const file = new File([blob], `${name.trim()}_photo.jpg`, { type: 'image/jpeg' });

      // Call registration API with file
      const result = await apiService.registerPersonFile(name.trim(), file);

      if (result.success) {
        setRegistrationResult(result);
        setCurrentStep(3);
        notification.success({
          message: 'Registration Successful',
          description: `${name} has been registered successfully!`
        });
      } else {
        throw new Error(result.message || 'Registration failed');
      }

    } catch (error) {
      console.error('Registration error:', error);
      
      // Extract proper error message from API response
      let errorMessage = 'Registration failed';
      
      if (error.response?.data?.detail) {
        // Handle FastAPI validation errors
        if (Array.isArray(error.response.data.detail)) {
          // Extract first validation error message
          const firstError = error.response.data.detail[0];
          if (firstError && typeof firstError === 'object' && firstError.msg) {
            errorMessage = firstError.msg;
          } else if (firstError && typeof firstError === 'string') {
            errorMessage = firstError;
          } else {
            errorMessage = 'Validation error occurred';
          }
        } else if (typeof error.response.data.detail === 'string') {
          errorMessage = error.response.data.detail;
        } else {
          errorMessage = 'Server validation error';
        }
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      // Ensure errorMessage is always a string
      if (typeof errorMessage !== 'string') {
        errorMessage = 'An unexpected error occurred during registration';
      }
      
      setError(errorMessage);
      notification.error({
        message: 'Registration Failed',
        description: errorMessage
      });
    } finally {
      setLoading(false);
    }
  };

  // Reset form and start over
  const resetRegistration = () => {
    setCurrentStep(0);
    setCapturedImage(null);
    setUploadedFile(null);
    setRegistrationResult(null);
    setError(null);
    setWebcamError(null);
    setShowWebcam(false);
    setWebcamReady(false);
    setWebcamInitializing(false);
    form.resetFields();
  };

  // Retry capture
  const retryCapture = () => {
    setCapturedImage(null);
    setUploadedFile(null);
    setWebcamReady(false);
    setWebcamInitializing(true);
    if (registrationMethod === 'webcam') {
      setCurrentStep(1);
      setShowWebcam(true);
    } else {
      setCurrentStep(1);
      setShowWebcam(false);
    }
  };

  const uploadProps = {
    name: 'file',
    accept: process.env.REACT_APP_SUPPORTED_FORMATS || 'image/*',
    showUploadList: false,
    beforeUpload: () => false, // Prevent auto upload
    onChange: handleFileUpload,
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            👤 Register New Person
          </Title>
          <Text type="secondary">Add a new person to the face recognition system</Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={resetRegistration}>
            Start Over
          </Button>
        </Col>
      </Row>

      {/* Progress Steps */}
      <Card style={{ marginBottom: '24px' }}>
        <Steps current={currentStep} size="small">
          <Step title="Enter Details" icon={<EditOutlined />} />
          <Step title="Capture Photo" icon={<CameraOutlined />} />
          <Step title="Review & Submit" icon={<InfoCircleOutlined />} />
          <Step title="Complete" icon={<CheckCircleOutlined />} />
        </Steps>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Registration Error"
          description={
            typeof error === 'string' 
              ? error 
              : 'An unexpected error occurred. Please try again.'
          }
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          {/* Step 0: Enter Name and Choose Method */}
          {currentStep === 0 && (
            <Card title="Step 1: Person Details">
              <Form form={form} onFinish={handleStepOne} layout="vertical">
                <Form.Item
                  name="name"
                  label="Full Name"
                  rules={[
                    { required: true, message: 'Please enter the person\'s name' },
                    { min: 1, max: 100, message: 'Name must be between 1 and 100 characters' }
                  ]}
                >
                  <Input
                    size="large"
                    placeholder="Enter full name"
                    prefix={<UserAddOutlined />}
                  />
                </Form.Item>

                <Divider />

                <Text strong>Choose Registration Method:</Text>
                <div style={{ marginTop: '16px' }}>
                  <Space direction="vertical" size="middle">
                    <Button
                      type={registrationMethod === 'webcam' ? 'primary' : 'default'}
                      icon={<CameraOutlined />}
                      onClick={() => {
                        log('User clicked: Use Webcam');
                        setRegistrationMethod('webcam');
                      }}
                      block
                    >
                      Use Webcam (Recommended)
                    </Button>
                    <Button
                      type={registrationMethod === 'upload' ? 'primary' : 'default'}
                      icon={<UploadOutlined />}
                      onClick={() => {
                        log('User clicked: Upload Photo');
                        setRegistrationMethod('upload');
                      }}
                      block
                    >
                      Upload Photo
                    </Button>
                  </Space>
                </div>

                <div style={{ marginTop: '24px' }}>
                  <Button type="primary" htmlType="submit" size="large" block>
                    Continue
                  </Button>
                </div>
              </Form>
            </Card>
          )}

          {/* Step 1: Capture Photo from Webcam */}
          {currentStep === 1 && registrationMethod === 'webcam' && (
            <Card title="Step 2: Capture Photo">
              <div style={{ textAlign: 'center' }}>
                {!isSecureContext && (
                  <Alert
                    message="Security Notice"
                    description="Camera access may be restricted on non-HTTPS connections. If you have issues, try using file upload instead or access via HTTPS."
                    type="warning"
                    showIcon
                    style={{ marginBottom: '16px', textAlign: 'left' }}
                  />
                )}
                
                <Alert
                  message="Webcam Instructions"
                  description="1. Click 'Allow' when browser asks for camera permission. 2. Wait for camera to load. 3. Position your face in the frame. 4. Click 'Capture Photo'."
                  type="info"
                  showIcon
                  style={{ marginBottom: '16px', textAlign: 'left' }}
                />
                
                <div style={{ marginBottom: '16px' }}>
                  <Text>Position your face in the camera frame and click capture</Text>
                </div>
                
                <div style={{ 
                  display: 'inline-block',
                  border: '2px solid #f0f0f0',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  marginBottom: '16px',
                  position: 'relative'
                }}>
                  {webcamInitializing && (
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      zIndex: 10,
                      background: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '20px',
                      borderRadius: '8px',
                      textAlign: 'center'
                    }}>
                      <Spin size="large" />
                      <div style={{ marginTop: '10px' }}>Initializing Camera...</div>
                    </div>
                  )}
                  
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    onUserMedia={handleWebcamUserMedia}
                    onUserMediaError={handleWebcamUserMediaError}
                    style={{ 
                      width: '100%', 
                      maxWidth: '640px',
                      height: 'auto',
                      display: showWebcam ? 'block' : 'none'
                    }}
                  />
                  
                  {/* Status overlay */}
                  {webcamReady && (
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      right: '10px',
                      background: 'rgba(0,255,0,0.8)',
                      color: 'white',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      Camera Ready
                    </div>
                  )}
                  
                  {webcamError && (
                    <div style={{
                      position: 'absolute',
                      top: '10px',
                      right: '10px',
                      background: 'rgba(255,0,0,0.8)',
                      color: 'white',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      Camera Error
                    </div>
                  )}
                </div>

                <div>
                  {!webcamReady && !webcamInitializing && (
                    <div style={{ marginBottom: '16px' }}>
                      <Alert
                        message="Camera Loading"
                        description="Please allow camera access when prompted by your browser. If camera doesn't load after 10 seconds, try refreshing the page or check browser settings."
                        type="warning"
                        showIcon
                        action={
                          <Button size="small" type="link" onClick={() => window.location.reload()}>
                            Refresh Page
                          </Button>
                        }
                      />
                    </div>
                  )}
                  
                  {webcamError && (
                    <div style={{ marginBottom: '16px' }}>
                      <Alert
                        message="Camera Error"
                        description={webcamError}
                        type="error"
                        showIcon
                        action={
                          <Space>
                            <Button size="small" type="link" onClick={() => setRegistrationMethod('upload')}>
                              Use File Upload Instead
                            </Button>
                          </Space>
                        }
                      />
                    </div>
                  )}
                  
                  <Space>
                    <Button
                      type="primary"
                      size="large"
                      icon={<CameraOutlined />}
                      onClick={() => {
                        log('User clicked: Capture Photo button');
                        capturePhoto();
                      }}
                      disabled={!webcamReady || webcamInitializing}
                    >
                      {webcamInitializing ? 'Initializing...' : webcamReady ? 'Capture Photo' : 'Waiting for Camera...'}
                    </Button>
                    <Button onClick={() => setCurrentStep(0)}>
                      Back
                    </Button>
                    <Button 
                      onClick={() => {
                        setRegistrationMethod('upload');
                        setCurrentStep(1);
                      }}
                    >
                      Switch to Upload
                    </Button>
                  </Space>
                </div>
              </div>
            </Card>
          )}

          {/* Upload Photo Option */}
          {currentStep === 1 && registrationMethod === 'upload' && (
            <Card title="Step 2: Upload Photo">
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Upload {...uploadProps}>
                  <Button size="large" icon={<UploadOutlined />}>
                    Select Photo
                  </Button>
                </Upload>
                <div style={{ marginTop: '16px' }}>
                  <Text type="secondary">
                    Supported formats: JPG, PNG (Max size: 10MB)
                  </Text>
                </div>
                <div style={{ marginTop: '24px' }}>
                  <Button onClick={() => setCurrentStep(0)}>
                    Back
                  </Button>
                </div>
              </div>
            </Card>
          )}

          {/* Step 2: Review and Submit */}
          {currentStep === 2 && (
            <Card title="Step 3: Review and Submit">
              <Form form={form} onFinish={handleRegistration} layout="vertical">
                <div style={{ textAlign: 'center', marginBottom: '24px' }}>
                  <Text strong>Captured Photo:</Text>
                  <div style={{ margin: '16px 0' }}>
                    <Image
                      src={capturedImage}
                      alt="Captured"
                      style={{ 
                        maxWidth: '300px', 
                        border: '2px solid #f0f0f0',
                        borderRadius: '8px'
                      }}
                    />
                  </div>
                  
                  <Space>
                    <Button icon={<ReloadOutlined />} onClick={retryCapture}>
                      Retake Photo
                    </Button>
                  </Space>
                </div>

                <Form.Item
                  name="name"
                  label="Confirm Name"
                  rules={[
                    { required: true, message: 'Please confirm the person\'s name' },
                    { min: 1, max: 100, message: 'Name must be between 1 and 100 characters' }
                  ]}
                >
                  <Input size="large" placeholder="Confirm full name" />
                </Form.Item>

                <div style={{ marginTop: '24px' }}>
                  <Space>
                    <Button
                      type="primary"
                      size="large"
                      htmlType="submit"
                      loading={loading}
                      icon={<UserAddOutlined />}
                    >
                      Register Person
                    </Button>
                    <Button size="large" onClick={retryCapture}>
                      Back
                    </Button>
                  </Space>
                </div>
              </Form>
            </Card>
          )}

          {/* Step 3: Registration Complete */}
          {currentStep === 3 && registrationResult && (
            <Card title="Registration Complete" style={{ textAlign: 'center' }}>
              <div style={{ padding: '40px' }}>
                <CheckCircleOutlined 
                  style={{ fontSize: '64px', color: '#52c41a', marginBottom: '24px' }} 
                />
                
                <Title level={3} style={{ color: '#52c41a' }}>
                  Registration Successful!
                </Title>
                
                <Paragraph>
                  <strong>{form.getFieldValue('name')}</strong> has been successfully registered 
                  in the face recognition system.
                </Paragraph>

                {registrationResult.detection_score && (
                  <div style={{ marginBottom: '24px' }}>
                    <Text type="secondary">
                      Face Detection Confidence: {registrationResult.detection_score.toFixed(3)}
                    </Text>
                  </div>
                )}

                <Space size="large">
                  <Button type="primary" size="large" onClick={resetRegistration}>
                    Register Another Person
                  </Button>
                </Space>
              </div>
            </Card>
          )}
        </Col>

        {/* Instructions Panel */}
        <Col xs={24} lg={8}>
          <Card title="Registration Guidelines" style={{ position: 'sticky', top: '24px' }}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              {/* Debug Information */}
              <div style={{ background: '#f0f0f0', padding: '8px', borderRadius: '4px', fontSize: '12px' }}>
                <strong>Debug Info:</strong><br/>
                Current Step: {currentStep}<br/>
                Registration Method: {registrationMethod}<br/>
                Show Webcam: {showWebcam ? 'true' : 'false'}<br/>
                Webcam Ready: {webcamReady ? 'true' : 'false'}<br/>
                Initializing: {webcamInitializing ? 'true' : 'false'}
              </div>
              
              <div>
                <Title level={5}>📸 Photo Requirements</Title>
                <ul style={{ marginLeft: '16px', color: '#666' }}>
                  <li>Clear, well-lit face</li>
                  <li>Face should be centered</li>
                  <li>No sunglasses or face coverings</li>
                  <li>Look directly at camera</li>
                  <li>Neutral expression preferred</li>
                </ul>
              </div>

              <div>
                <Title level={5}>🎥 Camera Permissions</Title>
                <ul style={{ marginLeft: '16px', color: '#666' }}>
                  <li>Click "Allow" when browser asks for camera access</li>
                  <li>Make sure camera is not being used by other apps</li>
                  <li>Try refreshing the page if camera doesn't load</li>
                  <li>Check browser settings if camera access is blocked</li>
                </ul>
              </div>

              <Divider />

              <div>
                <Title level={5}>✅ Best Practices</Title>
                <ul style={{ marginLeft: '16px', color: '#666' }}>
                  <li>Use good lighting</li>
                  <li>Keep face still during capture</li>
                  <li>Ensure only one person in frame</li>
                  <li>Multiple angles can improve accuracy</li>
                </ul>
              </div>

              <Divider />

              <div>
                <Title level={5}>🔒 Privacy & Security</Title>
                <ul style={{ marginLeft: '16px', color: '#666' }}>
                  <li>Images are processed securely</li>
                  <li>Only face embeddings are stored</li>
                  <li>Data encrypted at rest</li>
                  <li>GDPR compliant processing</li>
                </ul>
              </div>

              {currentStep === 2 && (
                <>
                  <Divider />
                  <Alert
                    message="Ready to Submit"
                    description="Please review the captured photo and confirm the name before submitting."
                    type="info"
                    showIcon
                  />
                </>
              )}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Registration;