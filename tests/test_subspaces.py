import unittest
import numpy as np
from src.subspaces import angles, selective_basis, basis, null_overlaps, whitened_angles

class MathTests(unittest.TestCase):
    def test_planted_angles(self):
        for degrees in (0,15,45,75,90):
            theta=np.radians(degrees)
            a=np.array([[1.],[0.],[0.]])
            b=np.array([[np.cos(theta)],[np.sin(theta)],[0.]])
            measured,overlap=angles(a,b)
            self.assertAlmostEqual(measured[0],degrees,places=5)
            self.assertAlmostEqual(overlap,np.cos(theta)**2,places=5)
    def test_rank_deficient_selective(self):
        a=np.eye(5)[:, :2]
        t=np.eye(5)[:, :3]
        self.assertEqual(selective_basis(a,t).shape[1],0)
    def test_null_distribution(self):
        a=np.eye(30)[:,:2];b=np.eye(30)[:,2:4]
        sample=null_overlaps(a,b,repetitions=20,seed=12)
        self.assertEqual(sample.shape,(20,))
        self.assertTrue(np.all((sample>=0)&(sample<=1)))
    def test_singular_rank(self):
        x=np.zeros((4,10));x[:,1]=2
        u,s=basis(x)
        self.assertEqual(u.shape,(10,1))
    def test_whitened_bounded(self):
        rng=np.random.default_rng(4)
        a=rng.normal(size=(12,10));t=rng.normal(size=(12,10))
        th,v=whitened_angles(a,t)
        self.assertTrue(0<=v<=1.0000001)

if __name__=='__main__': unittest.main()
