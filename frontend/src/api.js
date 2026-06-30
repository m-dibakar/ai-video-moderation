import axios from 'axios';

const BASE_URL = 'http://localhost:8001';

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await axios.post(`${BASE_URL}/moderate`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const pollJob = async (jobId) => {
  const res = await axios.get(`${BASE_URL}/jobs/${jobId}`);
  return res.data;
};
