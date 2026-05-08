-- Run this SQL in Supabase SQL Editor to create the scheduled_screenings table

CREATE TABLE IF NOT EXISTS public.scheduled_screenings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    doctor_id UUID NOT NULL REFERENCES public.doctors(id) ON DELETE CASCADE,
    scheduled_date DATE NOT NULL,
    scheduled_time TIME DEFAULT '09:00',
    notes TEXT,
    status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.scheduled_screenings ENABLE ROW LEVEL SECURITY;

-- Create policy to allow doctors to see their own scheduled screenings
CREATE POLICY "Doctors can view their own scheduled screenings"
ON public.scheduled_screenings FOR SELECT
USING (doctor_id IN (SELECT id FROM doctors WHERE user_id = auth.uid()));

-- Create policy to allow doctors to insert their own scheduled screenings
CREATE POLICY "Doctors can insert their own scheduled screenings"
ON public.scheduled_screenings FOR INSERT
WITH CHECK (doctor_id IN (SELECT id FROM doctors WHERE user_id = auth.uid()));

-- Create policy to allow doctors to delete their own scheduled screenings
CREATE POLICY "Doctors can delete their own scheduled screenings"
ON public.scheduled_screenings FOR DELETE
USING (doctor_id IN (SELECT id FROM doctors WHERE user_id = auth.uid()));

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_scheduled_screenings_doctor_id ON public.scheduled_screenings(doctor_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_screenings_patient_id ON public.scheduled_screenings(patient_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_screenings_date ON public.scheduled_screenings(scheduled_date);
