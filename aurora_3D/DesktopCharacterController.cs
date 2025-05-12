// File: DesktopCharacterController.cs
using UnityEngine;

public class DesktopCharacterController : MonoBehaviour
{
    private Animator animator;
    private float moveSpeed = 1.0f; // Normal walk speed
    private float sprintSpeed = 2.0f; // Sprint speed multiplier
    private bool isCrouching = false;
    private bool isProne = false;
    private bool isGrounded = true;
    private float currentStandStyle = 0.8f; // Default idle pose
    private float rotationSpeed = 60f; // Degrees per second for turning

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        HandleInput();
        // Add to Update() in your movement script
        Vector3 moveDirection = new Vector3(0, 0, animator.GetFloat("VelocityZ")) * Time.deltaTime;
        transform.Translate(moveDirection * moveSpeed, Space.Self);
        UpdateAnimator();
    }

        void HandleInput()
    {
        float moveX = 0f;
        float moveZ = 0f;

        // Forward/backward (Z movement) - W/S keys
        if (Input.GetKey(KeyCode.W))
        {
            moveZ = 1f;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            moveZ = -1f;
        }

        // Left/right strafing (X movement) - A/D keys
        if (Input.GetKey(KeyCode.A))
        {
            moveX = -1f;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            moveX = 1f;
        }

        Vector3 moveVector = new Vector3(moveX, 0, moveZ).normalized;

        float speed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift))
        {
            speed = sprintSpeed;
        }

        moveVector *= speed;

        animator.SetFloat("VelocityX", moveVector.x);
        animator.SetFloat("VelocityY", 0f);
        animator.SetFloat("VelocityZ", moveVector.z);

        // Only rotate with forward/backward (W/S)
        if (moveZ != 0f)
        {
            transform.Rotate(Vector3.up, moveX * rotationSpeed * Time.deltaTime);
        }

        // Jump (fake jump and fall states)
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isGrounded = false;
            animator.SetBool("Grounded", isGrounded);
            // Simulate jump arc
            Invoke(nameof(Land), 0.8f); 
        }

        // Crouch toggle
        if (Input.GetKeyDown(KeyCode.C))
        {
            isCrouching = !isCrouching;
            isProne = false;
        }

        // Prone toggle
        if (Input.GetKeyDown(KeyCode.P))
        {
            isProne = !isProne;
            isCrouching = false;
        }

        // Change standing style
        if (Input.GetKeyDown(KeyCode.Alpha0)) currentStandStyle = 0f; // Default
        if (Input.GetKeyDown(KeyCode.Alpha1)) currentStandStyle = 0.1f; // Style 1
        if (Input.GetKeyDown(KeyCode.Alpha2)) currentStandStyle = 0.2f; // Style 2
        if (Input.GetKeyDown(KeyCode.Alpha3)) currentStandStyle = 0.3f; // Style 3
        if (Input.GetKeyDown(KeyCode.Alpha4)) currentStandStyle = 0.4f; // Style 4
        if (Input.GetKeyDown(KeyCode.Alpha5)) currentStandStyle = 0.5f; // Style 5
        if (Input.GetKeyDown(KeyCode.Alpha6)) currentStandStyle = 0.6f; // Style 6
        if (Input.GetKeyDown(KeyCode.Alpha7)) currentStandStyle = 0.7f; // Style 7
        if (Input.GetKeyDown(KeyCode.Alpha8)) currentStandStyle = 0.8f; // Style 8

    }

    void Land()
    {
        isGrounded = true;
        animator.SetBool("Grounded", isGrounded);
    }

    void UpdateAnimator()
    {
        if (animator)
        {
            animator.SetBool("Grounded", isGrounded);

            if (isProne)
                animator.SetInteger("LocomotionMode", 2); // Prone
            else if (isCrouching)
                animator.SetInteger("LocomotionMode", 1); // Crouching
            else
                animator.SetInteger("LocomotionMode", 0); // Standing

            animator.SetFloat("StandStyle", currentStandStyle);
        }
    }
}
