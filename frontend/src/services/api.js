import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT) || 30000;

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging and error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response?.status === 404) {
      console.warn('Resource not found');
    } else if (error.response?.status === 500) {
      console.error('Server error occurred');
    } else if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
    }
    
    return Promise.reject(error);
  }
);

// API Service Class
class ApiService {
  // Face Recognition Endpoints
  
  async registerPerson(name, imageData) {
    try {
      const response = await apiClient.post('/api/faces/register', {
        name,
        image_data: imageData
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async registerPersonFile(name, file) {
    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('file', file);

      const response = await apiClient.post('/api/faces/register-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async recognizeFaces(imageData) {
    try {
      const response = await apiClient.post('/api/faces/recognize', {
        image_data: imageData
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async recognizeFacesFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post('/api/faces/recognize-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async liveRecognition(imageData) {
    try {
      const response = await apiClient.post('/api/faces/live-recognize', {
        image_data: imageData
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getAllUsers() {
    try {
      const response = await apiClient.get('/api/faces/users');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getUser(userId) {
    try {
      const response = await apiClient.get(`/api/faces/users/${userId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async deleteUser(userId) {
    try {
      const response = await apiClient.delete(`/api/faces/users/${userId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async addFaceSample(userId, imageData) {
    try {
      const response = await apiClient.post(`/api/faces/users/${userId}/add-sample`, {
        image_data: imageData
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async trainModels() {
    try {
      const response = await apiClient.post('/api/faces/train-models');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getTrainingStatus() {
    try {
      const response = await apiClient.get('/api/faces/training-status');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async deleteTrainedModels() {
    try {
      const response = await apiClient.delete('/api/faces/models');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSystemStatus() {
    try {
      const response = await apiClient.get('/api/faces/system-status');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Enhanced Training Methods (from original notebook approach)
  
  async trainEnhancedModels(useSynthetic = true) {
    try {
      const response = await apiClient.post(`/api/faces/train-enhanced?use_synthetic=${useSynthetic}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getEnhancedTrainingStatus() {
    try {
      const response = await apiClient.get('/api/faces/enhanced-training-status');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Dashboard Endpoints
  
  async getDashboardStats() {
    try {
      const response = await apiClient.get('/api/dashboard/stats');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getRecognitionTrends(days = 7) {
    try {
      const response = await apiClient.get(`/api/dashboard/recognition-trends?days=${days}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getTopRecognizedUsers(limit = 10) {
    try {
      const response = await apiClient.get(`/api/dashboard/top-recognized-users?limit=${limit}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSystemPerformance() {
    try {
      const response = await apiClient.get('/api/dashboard/system-performance');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // System Health
  
  async healthCheck() {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Sign Detection API Methods

  async getSignDetectionStatus() {
    try {
      const response = await apiClient.get('/api/signs/status');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async detectSigns(requestData) {
    try {
      const response = await apiClient.post('/api/signs/detect', requestData);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSignStatistics() {
    try {
      const response = await apiClient.get('/api/signs/statistics');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSignDetectionLogs(limit = 100) {
    try {
      const response = await apiClient.get(`/api/signs/logs?limit=${limit}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getEmergencySignLogs(limit = 50) {
    try {
      const response = await apiClient.get(`/api/signs/emergency-logs?limit=${limit}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSupportedGestures() {
    try {
      const response = await apiClient.get('/api/signs/gestures');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async clearSignDetectionLogs() {
    try {
      const response = await apiClient.delete('/api/signs/logs');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Helper Methods

  handleError(error) {
    if (error.response) {
      // Server responded with error status
      let message = 'Server error occurred';
      
      if (error.response.data?.detail) {
        // Handle FastAPI validation errors
        if (Array.isArray(error.response.data.detail)) {
          const firstError = error.response.data.detail[0];
          if (firstError?.msg) {
            message = firstError.msg;
          } else if (typeof firstError === 'string') {
            message = firstError;
          }
        } else if (typeof error.response.data.detail === 'string') {
          message = error.response.data.detail;
        }
      } else if (error.response.data?.message) {
        message = error.response.data.message;
      }
      
      return {
        success: false,
        message: message,
        status: error.response.status,
        data: error.response.data
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        success: false,
        message: 'Network error - please check your connection',
        status: 0
      };
    } else {
      // Something else happened
      return {
        success: false,
        message: error.message || 'An unexpected error occurred',
        status: -1
      };
    }
  }

  // Utility function to convert file to base64
  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  }

  // Utility function to convert canvas to base64
  canvasToBase64(canvas, quality = 0.8) {
    return canvas.toDataURL('image/jpeg', quality);
  }

  // Utility function to validate image file
  validateImageFile(file) {
    const maxSize = parseInt(process.env.REACT_APP_MAX_FILE_SIZE) || 10485760; // 10MB
    const supportedFormats = process.env.REACT_APP_SUPPORTED_FORMATS?.split(',') || 
      ['image/jpeg', 'image/jpg', 'image/png'];

    if (!file) {
      return { valid: false, message: 'No file provided' };
    }

    if (file.size > maxSize) {
      return { 
        valid: false, 
        message: `File size too large. Maximum size: ${Math.round(maxSize / 1024 / 1024)}MB` 
      };
    }

    if (!supportedFormats.includes(file.type)) {
      return { 
        valid: false, 
        message: `Unsupported file format. Supported: ${supportedFormats.join(', ')}` 
      };
    }

    return { valid: true };
  }
}

// Create singleton instance
const apiServiceInstance = new ApiService();

// Export singleton instance
export default apiServiceInstance;