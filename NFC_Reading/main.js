import { createSSRApp } from 'vue'
import App from './App.uvue'

export function createApp() {
  const app = createSSRApp(App)
  return {
    app
  }
} 