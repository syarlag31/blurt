import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import './index.css'
import './styles/theme-light.css'
import './theme-dark.css'
import './styles/expandable.css'
import './styles/swipe-animations.css'
import './styles/touch-targets.css'
import router from './router'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
