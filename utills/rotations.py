import torch

class RotationVectors:
    def unit_vector(self, vectors):
        """
        TORCH
        Returns the unit vectors of the input vectors.
        vectors: torch tensor of shape (batch_size, vector_dim)
        """
        return vectors / torch.norm(vectors, dim=1, keepdim=True)

    def angle(self, vectors1, vectors2):
        """
        Returns the angles in radians between the given batches of vectors.
        vectors1, vectors2: torch tensors of shape (batch_size, vector_dim)
        """
        v1_u = self.unit_vector(vectors1)
        v2_u = self.unit_vector(vectors2)

        minor = torch.det(torch.stack((v1_u[:, -2:], v2_u[:, -2:]), dim=1))
        if (minor == 0).any():
            raise NotImplementedError('Some vectors are too odd!')
        return torch.sign(minor) * torch.acos(torch.clamp(torch.sum(v1_u * v2_u, dim=1), -1.0, 1.0))

    def get_rotation_matrix(self, vec_1, vec_2):
        """
        Compute rotation matrices between batches of vectors.
        vec_1, vec_2: torch tensors of shape (batch_size, vector_dim)
        """
        a = self.angle(vec_1, vec_2)
        n1 = self.unit_vector(vec_1)

        vec_2 = vec_2 - torch.sum(n1 * vec_2, dim=1, keepdim=True) * n1
        n2 = self.unit_vector(vec_2)

        # Assuming vec_1, n1, n2, and a are already defined as torch tensors
        I = torch.eye(vec_1.shape[1]).to(vec_1.device)
        R = I + ((n2.unsqueeze(2) @ n1.unsqueeze(1)) - (n1.unsqueeze(2) @ n2.unsqueeze(1))) * torch.sin(a).unsqueeze(1).unsqueeze(2) + \
            ((n1.unsqueeze(2) @ n1.unsqueeze(1)) + (n2.unsqueeze(2) @ n2.unsqueeze(1))) * (torch.cos(a).unsqueeze(1).unsqueeze(2) - 1)

        return R