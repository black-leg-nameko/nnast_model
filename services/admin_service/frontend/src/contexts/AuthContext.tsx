import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import type { User, AuthState } from '../types'
import { mockUser } from '../api/mockData'

interface AuthContextType extends AuthState {
  login: () => void
  logout: () => void
  loginWithGitHub: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Check if we're in demo mode (no real backend)
const IS_DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true' || !import.meta.env.VITE_API_BASE_URL

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
  })

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('jwt_token')
      
      if (IS_DEMO_MODE) {
        // In demo mode, check if user was "logged in"
        const demoLoggedIn = localStorage.getItem('demo_logged_in')
        if (demoLoggedIn) {
          setState({
            user: mockUser,
            isAuthenticated: true,
            isLoading: false,
          })
        } else {
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          })
        }
        return
      }

      if (!token) {
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        })
        return
      }

      try {
        // Verify token with backend
        const response = await fetch('/api/v1/auth/me', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })

        if (response.ok) {
          const user = await response.json()
          setState({
            user,
            isAuthenticated: true,
            isLoading: false,
          })
        } else {
          localStorage.removeItem('jwt_token')
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
          })
        }
      } catch {
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
        })
      }
    }

    checkAuth()
  }, [])

  const login = useCallback(() => {
    if (IS_DEMO_MODE) {
      // Demo mode: simulate login
      localStorage.setItem('demo_logged_in', 'true')
      setState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false,
      })
    }
  }, [])

  const loginWithGitHub = useCallback(() => {
    if (IS_DEMO_MODE) {
      // Demo mode: simulate GitHub OAuth
      localStorage.setItem('demo_logged_in', 'true')
      setState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false,
      })
      return
    }

    // Real GitHub OAuth flow
    const clientId = import.meta.env.VITE_GITHUB_CLIENT_ID
    const redirectUri = `${window.location.origin}/auth/callback`
    const scope = 'read:user user:email'
    
    const githubAuthUrl = `https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&scope=${scope}`
    window.location.href = githubAuthUrl
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('jwt_token')
    localStorage.removeItem('demo_logged_in')
    setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
    })
  }, [])

  return (
    <AuthContext.Provider
      value={{
        ...state,
        login,
        logout,
        loginWithGitHub,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
