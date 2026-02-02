import { describe, it, expect } from 'vitest'
import { toApiName } from './api'

describe('toApiName', () => {
  describe('special mappings', () => {
    it('maps "Search & Rescue" to "search_and_rescue"', () => {
      expect(toApiName('Search & Rescue')).toBe('search_and_rescue')
    })

    it('maps "Infrastructure" to "infrastructure_related"', () => {
      expect(toApiName('Infrastructure')).toBe('infrastructure_related')
    })

    it('maps "Other Infrastructure" to "other_infrastructure"', () => {
      expect(toApiName('Other Infrastructure')).toBe('other_infrastructure')
    })

    it('maps "Other Weather" to "other_weather"', () => {
      expect(toApiName('Other Weather')).toBe('other_weather')
    })

    it('maps "Other Aid" to "other_aid"', () => {
      expect(toApiName('Other Aid')).toBe('other_aid')
    })

    it('maps "Medical Help" to "medical_help"', () => {
      expect(toApiName('Medical Help')).toBe('medical_help')
    })

    it('maps "Medical Products" to "medical_products"', () => {
      expect(toApiName('Medical Products')).toBe('medical_products')
    })

    it('maps "Aid Centers" to "aid_centers"', () => {
      expect(toApiName('Aid Centers')).toBe('aid_centers')
    })

    it('maps "Child Alone" to "child_alone"', () => {
      expect(toApiName('Child Alone')).toBe('child_alone')
    })

    it('maps "Direct Report" to "direct_report"', () => {
      expect(toApiName('Direct Report')).toBe('direct_report')
    })

    it('maps "Missing People" to "missing_people"', () => {
      expect(toApiName('Missing People')).toBe('missing_people')
    })
  })

  describe('default transformation', () => {
    it('converts simple category names to lowercase with underscores', () => {
      expect(toApiName('Water')).toBe('water')
      expect(toApiName('Food')).toBe('food')
      expect(toApiName('Shelter')).toBe('shelter')
      expect(toApiName('Transport')).toBe('transport')
    })

    it('handles multi-word categories with default transformation', () => {
      expect(toApiName('Weather Related')).toBe('weather_related')
      expect(toApiName('Buildings')).toBe('buildings')
      expect(toApiName('Electricity')).toBe('electricity')
    })

    it('handles edge cases', () => {
      expect(toApiName('')).toBe('')
      expect(toApiName('   ')).toBe('___')
      expect(toApiName('MixedCase')).toBe('mixedcase')
    })
  })

  describe('regression prevention', () => {
    it('ensures Infrastructure maps correctly (regression test)', () => {
      // This test specifically prevents the bug where "Infrastructure" 
      // was falling back to "infrastructure" instead of "infrastructure_related"
      const result = toApiName('Infrastructure')
      expect(result).toBe('infrastructure_related')
      expect(result).not.toBe('infrastructure')
    })
  })
})
