// File: DesktopCharacterController.cs
using UnityEngine;

public class DesktopCharacterController : MonoBehaviour
{
    private Animator animator;
    private float moveSpeed = 1.3f; // Walk speed
    private float sprintSpeed = 2.0f; // Sprint speed
    private bool isCrouching = false;
    private bool isProne = false;
    private bool isGrounded = true;
    private float currentStandStyle = 0.8f; // Default idle pose
    private float rotationSpeed = 180f; // Rotation speed in degrees/sec

    void Start()
    {
        animator = GetComponent<Animator>();
    }

    void Update()
    {
        HandleInput();
        UpdateAnimator();
    }

    void HandleInput()
    {
        float moveX = 0f;
        float moveZ = 0f;

        // Movement input
        if (Input.GetKey(KeyCode.W))
            moveZ = 1f;
        else if (Input.GetKey(KeyCode.S))
            moveZ = -1f;

        if (Input.GetKey(KeyCode.A))
            moveX = -1f;
        else if (Input.GetKey(KeyCode.D))
            moveX = 1f;

        Vector3 moveVector = new Vector3(moveX, 0, moveZ).normalized;

        float currentSpeed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift))
        {
            currentSpeed = sprintSpeed;
        }

        moveVector *= currentSpeed;

        animator.SetFloat("VelocityX", moveVector.x);
        animator.SetFloat("VelocityY", 0f);
        animator.SetFloat("VelocityZ", moveVector.z);

        // Rotate character when moving forward/back
        if (moveZ != 0f)
        {
            transform.Rotate(Vector3.up, moveX * rotationSpeed * Time.deltaTime);
        }

        // Move the character based on input (manual move forward)
        transform.Translate(new Vector3(0, 0, animator.GetFloat("VelocityZ")) * Time.deltaTime * moveSpeed, Space.Self);

        // Jump (simulate ground up/down)
        if (Input.GetKeyDown(KeyCode.Space) && isGrounded)
        {
            isGrounded = false;
            animator.SetBool("Grounded", false);
            Invoke(nameof(Land), 0.8f);
        }

        // Toggle Crouch
        if (Input.GetKeyDown(KeyCode.C))
        {
            isCrouching = !isCrouching;
            isProne = false;
            UpdateLocomotionMode();
        }

        // Toggle Prone
        if (Input.GetKeyDown(KeyCode.P))
        {
            isProne = !isProne;
            isCrouching = false;
            UpdateLocomotionMode();
        }

        // Change Idle Style based on current mode
        if (Input.GetKeyDown(KeyCode.Alpha0)) SetIdleStyle(0f);
        if (Input.GetKeyDown(KeyCode.Alpha1)) SetIdleStyle(0.1f);
        if (Input.GetKeyDown(KeyCode.Alpha2)) SetIdleStyle(0.2f);
        if (Input.GetKeyDown(KeyCode.Alpha3)) SetIdleStyle(0.3f);
        if (Input.GetKeyDown(KeyCode.Alpha4)) SetIdleStyle(0.4f);
        if (Input.GetKeyDown(KeyCode.Alpha5)) SetIdleStyle(0.5f);
        if (Input.GetKeyDown(KeyCode.Alpha6)) SetIdleStyle(0.6f);
        if (Input.GetKeyDown(KeyCode.Alpha7)) SetIdleStyle(0.7f);
        if (Input.GetKeyDown(KeyCode.Alpha8)) SetIdleStyle(0.8f);
    }

// New method to apply style to correct mode
void SetIdleStyle(float style)
{
    if (isProne)
        animator.SetFloat("ProneStyle", style);
    else if (isCrouching)
        animator.SetFloat("CrouchStyle", style);
    else
        animator.SetFloat("StandStyle", style);

    currentStandStyle = style; // Save current style for later too
}

void UpdateLocomotionMode()
{
    if (animator == null) return;

    if (isProne)
    {
        animator.SetInteger("LocomotionMode", 2); // 2 = Prone
        animator.SetFloat("Upright", 0f);          // 0 = fully prone
        animator.SetFloat("ProneStyle", 0.8f); // Set a default prone pose (0-0.8)
        animator.SetFloat("CrouchStyle", 0f); // Force CrouchStyle to idle
        animator.SetFloat("StandStyle", 0f); // Force StandStyle to idle
    }
    else if (isCrouching)
    {
        animator.SetInteger("LocomotionMode", 1); // 1 = Crouch
        animator.SetFloat("Upright", 0.5f);          // 0 = fully prone
        animator.SetFloat("CrouchStyle", 0.8f); // Set a default crouch pose (0-0.8)
        animator.SetFloat("ProneStyle", 0); // Force ProneStyle to idle
        animator.SetFloat("StandStyle", 0f); // Force StandStyle to idle
    }
    else
    {
        animator.SetInteger("LocomotionMode", 0); // 0 = Standing
        animator.SetFloat("Upright", 1f);          // 0 = fully prone
        animator.SetFloat("StandStyle", 0.8f); // Default stand pose
        animator.SetFloat("CrouchStyle", 0f); // Idle crouch
        animator.SetFloat("ProneStyle", 0f); // Idle prone
    }
}

    void Land()
    {
        isGrounded = true;
        animator.SetBool("Grounded", true);
    }

    void UpdateAnimator()
    {
        if (animator)
        {
            animator.SetBool("Grounded", isGrounded);

            if (isProne)
                animator.SetInteger("LocomotionMode", 2);
            else if (isCrouching)
                animator.SetInteger("LocomotionMode", 1);
            else
                animator.SetInteger("LocomotionMode", 0);
        }
    }
}
